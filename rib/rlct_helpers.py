import functools
from typing import TYPE_CHECKING, Callable, Union

import pandas as pd
import torch
import torch.nn as nn

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import estimate_learning_coeff_with_summary

if TYPE_CHECKING:   # Prevent circular import to import type annotations
    from experiments.relu_interactions.rlct_estimation import Config


def estimate_rlcts_training(
    model: nn.Module,
    config: "Config",
    criterion: Callable,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[Union[float, pd.DataFrame], ...]:
    """Estimate RLCT for a given epoch during training.

    Needs to be called in same execution logic as eval_and_save() so that the logger uses the right step.
    Currently implemented for distillation training in main code.
    """
    num_samples = len(dataloader.dataset)
    config.rlct_config.sgld_kwargs["num_samples"] = config.rlct_config.sgnht_kwargs["num_samples"] = num_samples
    # Make a copy of current model state dict - used in sample() function
    checkpoint = {"state dict": model.state_dict()}

    rlct_func = functools.partial(
        estimate_learning_coeff_with_summary,
        model=model,
        loader=dataloader,
        criterion=criterion,
        config=config,
        checkpoint=checkpoint,
        num_draws=config.rlct_config.num_draws,
        num_chains=config.rlct_config.num_chains,
        num_burnin_steps=config.rlct_config.num_burnin_steps,
        num_steps_bw_draws=config.rlct_config.num_steps_bw_draws,
        cores=config.rlct_config.cores,
        seed=config.rlct_config.seed,
        pbar=config.rlct_config.pbar,
        device=device,
        verbose=config.rlct_config.verbose,
        use_distill_loss=config.rlct_config.use_distill_loss,
        sample_layer=config.rlct_config.sample_layer,
    )

    sgld_results_dict = rlct_func(
        sampling_method=SGLD, optimizer_kwargs=config.rlct_config.sgld_kwargs)
    sgnht_results_dict = rlct_func(
        sampling_method=SGNHT, optimizer_kwargs=config.rlct_config.sgnht_kwargs)

    sgld_mean, sgld_std = sgld_results_dict["mean"], sgld_results_dict["std"]
    sgnht_mean, sgnht_std = sgnht_results_dict["mean"], sgnht_results_dict["std"]
    trace_sgld, trace_sgnht = sgld_results_dict["trace"], sgnht_results_dict["trace"]

    return sgld_mean, sgld_std, sgnht_mean, sgnht_std, trace_sgld, trace_sgnht
