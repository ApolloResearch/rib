import itertools
from copy import deepcopy
from typing import Callable, Dict, Literal, Optional, Type, Union, TYPE_CHECKING

import pandas as pd
import torch
from torch import nn
from torch.multiprocessing import cpu_count, get_context
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgld_ma import SGLD_MA
from devinterp.optim.sgnht import SGNHT
from rib.models.utils import get_model_attr

if TYPE_CHECKING:   # Prevent circular import to import type annotations
    from experiments.relu_interactions.rlct_estimation import Config


def predictive_kl_loss(x: torch.Tensor, y: torch.Tensor, teacher_model: nn.Module, student_model: nn.Module, temperature: float = 1., **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard knowledge distillation loss (equivalent to KL divergence between the tempered teacher and student softmax outputs).
    """
    with torch.no_grad():
        teacher_logits = teacher_model(x)
    student_logits = student_model(x)
    return (
        nn.functional.kl_div(
            nn.functional.log_softmax(fix_transformer_logit(student_logits) / temperature, dim=-1),
            nn.functional.log_softmax(fix_transformer_logit(teacher_logits) / temperature, dim=-1),
            log_target=True,
            reduction="batchmean",
        ),
        student_logits,
    )


def fix_transformer_logit(logits: tuple) -> torch.Tensor:
    """Get rid of tuple and token dimension."""
    return rearrange(logits[0], 'b () d -> b d')


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    pbar: bool = False,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
    return_weights: bool = False,
    use_distill_loss: bool = False,
    sample_layer: Optional[str] = None,
) -> pd.DataFrame:
    """Instantiate a new model and optimizer for this chain and run sampling to get RLCT estimate.

    Args:
        sample_layer: Indicates which layer to calculate RLCT for. Everything else is frozen.
            If this parameter is None, all layers are sampled.
            MLP: should be e.g. "layers.0.activation"
            SequentialTransformer: should be e.g. "sections.section_0.0"
    Returns:
        local_draws: A DataFrame containing loss values with loss over steps.
            Save weights if returning weights is true. Also save acceptance ratio if using Metropolis step.
    """
    model, baseline_model = deepcopy(ref_model).to(device), deepcopy(ref_model).to(device)

    # Only pass parameters of layers that should not be frozen to the optimizer
    if sample_layer is None:
        optimizer_params = model.parameters()
    else:
        module = get_model_attr(model, sample_layer)
        # Prevent ambiguity of which modules are used
        assert not isinstance(module, torch.nn.ModuleList), "Layer to calculate RLCT must be a single module, not a ModuleList"
        optimizer_params = module.parameters()

    optimizer = sampling_method(optimizer_params, **(optimizer_kwargs.dict() or {}))

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    local_draws = pd.DataFrame(
        index=range(num_draws),
        columns=["chain", "step", "loss"] + (["model_weights"] if return_weights else []) + (
            ["accept_ratio"] if sampling_method == SGLD_MA else [])
    )

    iterator = zip(range(num_steps), itertools.cycle(loader))

    if pbar:
        iterator = tqdm(
            # TODO: Redundant
            iterator, desc=f"Chain {chain}", total=num_steps, disable=not verbose
        )

    model.train()

    for i, (xs, ys) in iterator:

        def closure(backward=True):
            """
            Compute loss for the current state of the model and update the gradients.

            Args:
                backward: Whether to perform backward pass. Only used for updating weight grad at proposed location. See SGLD_MA.step() for more details.
            """
            y_preds = model(xs)
            if use_distill_loss:
                loss, student_logits = predictive_kl_loss(
                    x=xs,
                    y=ys,
                    teacher_model=baseline_model,
                    student_model=model,
                    is_lm=is_lm,
                )
            else:
                y_preds = fix_transformer_logit(y_preds) if is_lm else y_preds
                loss = criterion(y_preds, ys)
            if backward:
                optimizer.zero_grad()
                loss.backward()
            return loss

        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs)
        is_lm: bool = True if y_preds[0].dim() == 3 else False

        if isinstance(y_preds, tuple): # For LM output can be tuple with empty second element
            y_preds = rearrange(y_preds[0], "b p logit_dim -> (b p) logit_dim")

        if use_distill_loss:
            loss, student_logits = predictive_kl_loss(
                x=xs,
                y=ys,
                teacher_model=baseline_model,
                student_model=model,
                is_lm=is_lm,
            )
        else:
            y_preds = fix_transformer_logit(y_preds) if is_lm else y_preds
            loss = criterion(y_preds, ys)

        loss.backward()

        if sampling_method in [SGLD, SGNHT]:
            optimizer.step(closure=None)
        elif sampling_method in [SGLD_MA]:
            optimizer.step(closure=closure)

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw_idx = (i - num_burnin_steps) // num_steps_bw_draws
            local_draws.loc[draw_idx, "step"] = i
            local_draws.loc[draw_idx, "chain"] = chain
            local_draws.loc[draw_idx, "loss"] = loss.detach().item()
            if return_weights:
                local_draws.loc[draw_idx, "model_weights"] = (
                    model.state_dict()["weights"].clone().detach()
                )

    if sampling_method == SGLD_MA:
        local_draws["accept_ratio"] = optimizer.accepted_updates / \
            optimizer.total_updates

    return local_draws


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    config: "Config",
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        step (Literal['sgld']): The name of the optimizer to use to step.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        progressbar (bool): Whether to display a progress bar.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
    """
    seed, num_chains, cores = config.rlct_config.seed, config.rlct_config.num_chains, config.rlct_config.cores

    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            num_draws=config.rlct_config.num_draws,
            num_burnin_steps=config.rlct_config.num_burnin_steps,
            num_steps_bw_draws=config.rlct_config.num_steps_bw_draws,
            pbar=config.rlct_config.pbar,
            device=device,
            verbose=config.rlct_config.verbose,
            use_distill_loss=config.rlct_config.use_distill_loss,
            sample_layer=config.rlct_config.sample_layer,
        )

    results = []

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [
                               get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))

    results_df = pd.concat(results, ignore_index=True)

    if sampling_method == SGLD_MA:
        for i, result in enumerate(results):
            results_df.loc[results_df['chain'] == i,
                           'accept_ratio'] = result["accept_ratio"].iloc[0]

    return results_df
