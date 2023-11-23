from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
import torch
import yaml
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor, nn

from rib.log import logger
from rib.utils import find_root

T = TypeVar("T")


ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


def save_model(config_dict: dict[str, Any], save_dir: Path, model: nn.Module, epoch: int) -> None:
    # If the save_dir doesn't exist, create it and save the config
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        logger.info("Saving config to %s", save_dir)
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
    logger.info("Saving model to %s", save_dir)
    torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch + 1}.pt")


def get_model_attr(model: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    """Retrieve a nested attribute of a PyTorch module by a string of attribute names.

    Each attribute name in the path is separated by a period ('.').

    Since models often have lists of modules, the attribute path can also include an index.

    Args:
        model (torch.nn.Module): The PyTorch model.
        attr_path (str): A string representing the path to the attribute.

    Returns:
        torch.nn.Module: The attribute of the model.

    Example:
        >>> mlp = MLP([5], input_size=2, output_size=3)
        >>> mlp
        MLP(
            (layers): ModuleList(
                (0): Layer(
                    (linear): Linear(in_features=2, out_features=5, bias=True)
                    (activation): ReLU()
                )
                (1): Layer(
                    (linear): Linear(in_features=5, out_features=3, bias=True)
                )
            )
        )
        - get_model_attr(model, "layers") -> ModuleList(...)
        - get_model_attr(model, "layers.0") -> Layer(...)
        - get_model_attr(model, "layers.0.activation") -> ReLU()
        - get_model_attr(model, "layers.1.linear") -> LinearFoldedBias(...)
    """
    attr_names = attr_path.split(".")
    attr = model

    for name in attr_names:
        try:
            if isinstance(attr, torch.nn.ModuleList) and name.isdigit():
                attr = attr[int(name)]
            else:
                attr = getattr(attr, name)
        except AttributeError:
            logger.error(f"Attribute '{name}' not found in the path '{attr_path}'.")
            raise
    return attr


def create_list_partitions(in_list: list[T], sub_list: list[T]) -> list[list[T]]:
    """Create partitions of a list based on a sub-list of matching values

    Args:
        in_list: The list to partition.
        sub_list: The sub-list of values to partition by.

    Returns:
        A list of lists, where each sub-list is a partition of the input list.

    Example:
        >>> all_layers = ['embed', 'pos_embed', 'add_embed', 'ln1.0', 'attn.0', 'add_resid1.0']
        >>> node_layers = ['ln1.0', 'add_resid1.0']
        >>> create_list_partitions(all_layers, node_layers)
        [['embed', 'pos_embed', 'add_embed'], ['ln1.0', 'attn.0'], ['add_resid1.0']]
    """
    indices: list[int] = []
    for entry in sub_list:
        assert entry in in_list, f"Entry '{entry}' not found in the input list."
        indices.append(in_list.index(entry))

    partitions: list[list[T]] = []
    for i, j in zip([0] + indices, indices + [None]):
        sub_list = in_list[i:j]
        if sub_list:
            partitions.append(sub_list)
    return partitions


def gelu_new(input: Float[Tensor, "... d_mlp"]) -> Float[Tensor, "... d_mlp"]:
    """Implementation of GeLU used by GPT2 - subtly different from PyTorch's.

    Taken from transformer-lens.
    """
    return (
        0.5
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def concat_zeros(x: Float[Tensor, "a b"]) -> None:
    """Concatenates zeros to the last dimension of a tensor. Updating the data in-place."""
    x.data = torch.cat(
        [x.data, torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1
    )  # (a, b+1)


def concat_ones(x: Float[Tensor, "a b"]) -> None:
    """Concatenates ones to the last dimension of a tensor. Updating the data in-place."""
    x.data = torch.cat(
        [x.data, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1
    )  # (a, b+1)


def fold_attn_QK(
    weight: Float[Tensor, "head_index d_model d_head"], bias: Float[Tensor, "head_index d_head"]
) -> None:
    """Concatenate the bias to the d_model dimension of the weight matrix and zero out the bias.

    This is used for the Q and K matrices in the attention layer.
    """
    weight.data = torch.cat(
        [weight.data, bias.data[:, None]], dim=1
    )  # (head_idx, d_model+1, d_head)
    bias.data = torch.zeros_like(bias.data)


def fold_attn_V(
    weight: Float[Tensor, "head_index d_model d_head"], bias: Float[Tensor, "head_index d_head"]
) -> None:
    """Fold in the bias to the W_V matrix.

    We concat the bias vector to 'row' (d_model) dimension and then add an extra
    'column' (d_head) dimension with all zeros and a single 1. I.e. W_V_folded will be of
    shape (n_head, d_model + 1, d_head + 1).

    This is used for the V matrices in the attention layer.
    """
    weight.data = torch.cat(
        [
            torch.cat([weight.data, bias.data[:, None, :]], dim=1),
            torch.cat(
                [
                    torch.zeros(
                        weight.shape[0],
                        weight.shape[1],
                        1,
                        device=weight.device,
                        dtype=weight.dtype,
                    ),
                    torch.ones(weight.shape[0], 1, 1, device=weight.device, dtype=weight.dtype),
                ],
                dim=1,
            ),
        ],
        dim=2,
    )  # (n_head, d_model+1, d_head+1)
    bias.data = torch.zeros(
        bias.shape[0], bias.shape[1] + 1, device=bias.device, dtype=bias.dtype
    )  # (n_head, d_head+1)


def fold_attn_O(
    weight: Float[Tensor, "head_index d_head d_model"], bias: Float[Tensor, "d_model"]
) -> None:
    """Fold in the b_O bias to the W_O matrix.

    Notice that there exists only one b_O vector (size d_model) independent of the number of
    attention heads. However, we want to fold in the bias into the (n_head) W_O matrices. To do this
    we first duplicate the b_O bias n_head times and multiply by 1/n_head (split_bias_data),
    essentially folding a bit of the bias into every W_O matrix.

    Then we concat the modified b_O and W_O via the 'row' (d_head) dimension, and add an extra
    'column' (d_model) of all zeros to match the folded resid shape. Note that this final extra
    column does have only zeros and no one(s), just like the W_out of the folded MLP, because the
    residual stream will already have an extra dimension of ones.

    """
    # Split up b_O over the different heads to distribute it over the W_O matrices
    # b_O current shape = (d_model,)
    # split_bias_data target shape = (n_head, d_model)
    split_bias_data = einsum(
        "n_head, d_model -> n_head d_model",
        torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype) / weight.shape[0],
        bias.data,
    )  # (n_head, d_model)
    # weight.data current shape = (n_head, d_head, d_model)
    weight.data = torch.cat(
        [
            torch.cat(
                [weight.data, split_bias_data[:, None, :]], dim=1
            ),  # (n_head, d_head+1, d_model)
            torch.zeros(
                weight.shape[0], weight.shape[1] + 1, 1, device=weight.device, dtype=weight.dtype
            ),  # (n_head, d_head+1, 1)
        ],
        dim=2,
    )  # (n_head, d_head+1, d_model+1)
    bias.data = torch.zeros(bias.shape[0] + 1, device=bias.device, dtype=bias.dtype)  # (d_model+1)


def fold_mlp_in(
    act_fn: Optional[str],
    weight: Float[Tensor, "d_model d_mlp"],
    bias: Float[Tensor, "d_mlp"],
) -> None:
    """Fold in the bias to the input weight matrix of the MLP.

    We concat the bias vector to the row dimension and then add an extra column with all zeros and
    a single value at the end equal to the value that would be transformed to 1 by the activation
    function (denoted root_1). I.e. W_in_folded will be of shape (d_model + 1, d_mlp + 1).

    If the activation function has a root_1 of 1 (such as ReLU, or the idenity/no activation fn), the value added to the end will be
    1. This will result in act_fn(x @ W_in_folded) giving the same result as act_fn(x @ W_in + b_in), except with an extra 1 concatenated at the end.
    """

    if act_fn == "relu" or act_fn is None:
        # relu(1) = 1
        root_one = 1.0
    elif act_fn in ["gelu", "gelu_new"]:
        gelu_fn = gelu_new if act_fn == "gelu_new" else torch.nn.functional.gelu  # type: ignore
        # Find the value of x such that act_fn(x) = 1
        root_one = find_root(
            lambda x: gelu_fn(x) - 1.0,  # type: ignore
            xmin=torch.tensor(-1.0),
            xmax=torch.tensor(4.0),
            tol=1e-11,
            max_iter=500,
        ).to(weight.dtype)
    else:
        raise ValueError(f"Unsupported activation function: {act_fn} for bias folding.")

    weight.data = torch.cat(
        [
            torch.cat([weight.data, bias.data[None, :]], dim=0),
            torch.cat(
                [
                    torch.zeros(weight.shape[0], 1, device=weight.device, dtype=weight.dtype),
                    torch.ones(1, 1, device=weight.device, dtype=weight.dtype) * root_one,
                ],
                dim=0,
            ),
        ],
        dim=1,
    )  # (d_model+1, d_mlp+1)
    bias.data = torch.zeros(bias.shape[0] + 1, device=bias.device, dtype=bias.dtype)  # (d_mlp+1)


def fold_mlp_out(weight: Float[Tensor, "d_mlp d_model"], bias: Float[Tensor, "d_model"]) -> None:
    """Fold in the bias to the output weight matrix of the MLP.

    We concat the bias vector to the row dimension and then add an extra column of all zeros.
    Since the MLP block is added to the residual stream, which already has a feature of ones
    concatenated to the end of it, adding both of these together will result in a feature of ones,
    which is what is needed to preserve the bias in the next layer.
    """
    weight.data = torch.cat(
        [
            torch.cat([weight.data, bias.data[None, :]], dim=0),
            torch.zeros(weight.shape[0] + 1, 1, device=weight.device, dtype=weight.dtype),
        ],
        dim=1,
    )  # (d_mlp+1, d_model+1)
    bias.data = torch.zeros(bias.shape[0] + 1, device=bias.device, dtype=bias.dtype)  # (d_model+1)


def fold_unembed(weight: Float[Tensor, "d_model d_vocab"], bias: Float[Tensor, "d_vocab"]) -> None:
    """Fold in the bias to the 0th dimension of the weight matrix and zero out the bias."""
    weight.data = torch.cat([weight.data, bias.data[None, :]], dim=0)  # (d_model+1, d_vocab)
    bias.data = torch.zeros(1, bias.shape[0], device=bias.device, dtype=bias.dtype)  # (1, d_vocab)


def layer_norm(x: Float[Tensor, "... d_model"], epsilon=1e-5) -> Float[Tensor, "... d_model"]:
    in_dtype = x.dtype
    if in_dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)
    x = x - x.mean(-1, keepdim=True)
    scale: Float[Tensor, "... 1"] = (x.pow(2).mean(-1, keepdim=True) + epsilon).sqrt()
    x = x / scale
    return x.to(in_dtype)
