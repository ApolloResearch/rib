import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Type, TypeVar, Union

import numpy as np
import torch
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from rib.hook_manager import Hook, HookedModel

T = TypeVar("T", bound=BaseModel)

REPO_ROOT = Path(__file__).parent.parent


@torch.inference_mode()
def eval_model_accuracy(
    hooked_model: "HookedModel",
    dataloader: DataLoader,
    hooks: Optional[list["Hook"]] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> float:
    """Run the model on the dataset and return the accuracy.

    In the case where the output has a position dimension, we take the last position as the output.

    Args:
        hooked_model: The model to evaluate.
        dataloader: The dataloader for the dataset.
        hooks: The hooks to use.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        device: The device to run the model on.

    Returns:
        The accuracy of the model on the dataset.
    """

    correct_predictions: int = 0
    total_predictions: int = 0

    for batch in dataloader:
        data, labels = batch
        data, labels = data.to(device=device), labels.to(device)
        # Change the dtype unless the inputs are integers (e.g. like they are for LMs)
        if data.dtype not in [torch.int64, torch.int32]:
            data = data.to(dtype=dtype)
        raw_output: Union[
            Float[Tensor, "batch d_vocab"], tuple[Float[Tensor, "batch pos d_vocab"]]
        ] = hooked_model(data, hooks=hooks)
        if isinstance(raw_output, tuple):
            assert len(raw_output) == 1, "Only one output is supported."
            # Check if the pos is 1, if so, squeeze it out. (This is the case for modular addition)
            output: Float[Tensor, "... d_vocab"] = raw_output[0]
            if output.ndim == 3:
                output = output[:, -1, :]
        else:
            output = raw_output

        # Assuming output is raw logits and labels are class indices.
        predicted_labels: Int[Tensor, "batch"] = output.argmax(dim=-1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.shape[0]

    accuracy: float = correct_predictions / total_predictions
    return accuracy


def load_config(config_path: Path, config_model: Type[T]) -> T:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = config_model(**config_dict)
    return config


def overwrite_output(out_file: Path) -> bool:
    """Check if the user wants to overwrite the output file."""
    response = input(f"Output file {out_file} already exists. Overwrite? (y/n) ")
    return response.lower() == "y"


def calc_ablation_schedule(
    ablate_every_vec_cutoff: Optional[int],
    n_vecs: int,
) -> list[int]:
    """Create a schedule for the number of vectors to ablate.

    The schedule is exponential with a base of 2, with the exception that from
    `ablate_every_vec_cutoff` to `n_vecs` we ablate every vector. The schedule also includes a run
    with no ablations.

    Args:
        ablate_every_vec_cutoff: The point in which we ablate every vector. If None, we ablate
        every vector in the schedule individually (i.e. no exponential schedule).

    Returns:
        The schedule for the number of vectors to ablate.

    Examples:
        >>> calc_ablation_schedule(None, 12)
        [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> calc_ablation_schedule(0, 12)
        [12, 11, 9, 5, 0]  # Exponential schedule (2^x) from the beginning.
        >>> calc_ablation_schedule(1, 12)
        [12, 11, 10, 8, 4, 0]  # Exponential schedule (2^x) after the first 1 value
        >>> calc_ablation_schedule(3, 12)
        [12, 11, 10, 9, 8, 6, 2, 0]
        >>> calc_ablation_schedule(3, 24)
        [24, 23, 22, 21, 20, 18, 14, 6, 0]
    """
    if ablate_every_vec_cutoff is None:
        return list(range(n_vecs, -1, -1))

    assert ablate_every_vec_cutoff < n_vecs, "ablate_every_vec_cutoff must be smaller than n_vecs"
    assert ablate_every_vec_cutoff >= 0, "ablate_every_vec_cutoff must be positive"
    # The section in which we ablate every vector.
    ablate_every_vecs: list[int] = list(range(n_vecs, n_vecs - ablate_every_vec_cutoff - 1, -1))
    # The section in which we ablate according to 2^x.
    ablate_exponential: list[int] = []
    prev_val = ablate_every_vecs[-1]
    for x in range(n_vecs):
        exp_val = prev_val - 2**x
        if exp_val > 0:
            ablate_exponential.append(exp_val)
            prev_val = exp_val
        else:
            # No more values to append, just add the case for no ablation and exit
            ablate_exponential.append(0)
            break

    # combine the two sections
    schedule = ablate_every_vecs + ablate_exponential
    assert schedule[0] == n_vecs, "The first element of the schedule must be n_vecs."
    assert schedule[-1] == 0, "The last element of the schedule must be 0."
    return schedule


def set_seed(seed: int = 0) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_root(
    func: Callable,
    xmin: Union[Float[Tensor, "1"], float] = torch.tensor(-1.0),
    xmax: Union[Float[Tensor, "1"], float] = torch.tensor(1.0),
    tol: float = 1e-6,
    max_iter: int = 100,
):
    """Find the root of a function using bisection."""

    # check that func(xmin) and func(xmax) have opposite signs
    assert func(xmin) * func(xmax) < 0, "func(xmin) and func(xmax) must have opposite signs"

    for i in range(max_iter):
        xmid = (xmin + xmax) / 2
        if func(xmid) * func(xmin) < 0:
            xmax = xmid
        else:
            xmin = xmid
        if abs(func(xmid)) < tol:
            return xmid
    raise ValueError(f"Finding the root of {func} via bisection failed to converge")
