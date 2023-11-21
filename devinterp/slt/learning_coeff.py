from typing import Callable, Dict, List, Literal, Optional, Type, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgld_ma import SGLD_MA
from devinterp.slt.sampler import sample

if TYPE_CHECKING:   # Prevent circular import to import type annotations
    from experiments.relu_interactions.rlct_estimation import Config


def estimate_learning_coeff(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    config: "Config",
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Use lambda hat equation and average and baseline loss."""
    trace: pd.DataFrame = sample(
        model=model,
        loader=loader,
        criterion=criterion,
        config=config,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )
    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    avg_loss = trace.groupby("chain")["loss"].mean().mean()
    num_samples = len(loader.dataset)

    if 'accept_ratio' in trace.columns:
        accept_ratio = trace.groupby("chain")["accept_ratio"].mean().mean()
        print(f"Acceptance ratio: {accept_ratio}")

    return (avg_loss - baseline_loss) * num_samples / np.log(num_samples)


def estimate_learning_coeff_with_summary(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    config: "Config",
    sampling_method: Type[torch.optim.Optimizer],
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Calls on sample function and returns summary mean, standard deviation, trace, and acceptance ratio for MH.

    Currently uses deepcopy() in sampler, but it might be better practise to instantiate new model
    and put state dict in.

    Args:
        sample_layer: If a list of ints, freezes the layers not indicated by these layers..
            If a dict, freezes the layers named in value list for the key module name.
    """
    trace = sample(
        model=model,
        loader=loader,
        criterion=criterion,
        config=config,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )

    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    num_samples = optimizer_kwargs.num_samples
    avg_losses = trace.groupby("chain")["loss"].mean()
    results = torch.zeros(config.rlct_config.num_chains, device=device)
    if sampling_method == SGLD_MA:
        accept_ratio = trace.groupby("chain")["accept_ratio"].mean().mean()
        print(f"Acceptance ratio: {accept_ratio}")
    else:
        accept_ratio = None

    for i in range(config.rlct_config.num_chains):
        chain_avg_loss = avg_losses.iloc[i]
        results[i] = (chain_avg_loss - baseline_loss) * \
            num_samples / np.log(num_samples)

    avg_loss = results.mean()
    std_loss = results.std()

    return {
        "mean": avg_loss.item(),
        "std": std_loss.item(),
        **{f"chain_{i}": results[i].item() for i in range(config.rlct_config.num_chains)},
        "trace": trace,
        "accept_ratio": accept_ratio,
    }


def plot_learning_coeff_trace(trace: pd.DataFrame, **kwargs):
    import matplotlib.pyplot as plt

    for chain, df in trace.groupby("chain"):
        plt.plot(df["step"], df["loss"], label=f"Chain {chain}", **kwargs)

    plt.xlabel("Step")
    plt.ylabel(r"$L_n(w)$")
    plt.title("Learning Coefficient Trace")
    plt.legend()
    plt.show()
