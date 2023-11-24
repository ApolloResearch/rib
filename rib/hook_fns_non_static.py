"""Contains hook functions whose only purpose is to change the model's activations on forward pass,
not store data."""
from copy import deepcopy
from typing import Any, Union

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor


def relu_swap_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    replacement_idxs: Int[Tensor, "d_hidden"],
    use_residual_stream: bool,
) -> Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ]:
    """Swap each ReLU with another ReLU. Whether the swap is based on clustering or otherwise
    depends on the indices you pass in.

    Hook activation layers.

    This function has to work with all possible modules, including those that don't have tuple as
    output. By default it is easier to work with the residual stream being present even if it is not
    modified.
    Hidden dimension of operators and intermediate tensor in this function should be d_resid + d_hidden_MLP.

    Args:
        module: Module whose activations are being hooked.
        inputs: Inputs to the module.
        output: Output of the module.
        hooked_data: Dictionary containing the hook data.
        hook_name: Name of the hook.
        data_key: Key of the data to be stored.
        replacement_idxs: Indices to replace the original indices with.
        use_residual_stream: Whether residual stream is included in similarity metrics and
        clustering and subsequent replacement. Default False until I figured out whether this is principled.
    """
    output_is_tuple = True if isinstance(output, tuple) else False
    is_lm = True if inputs[0].dim() == 3 else False
    outputs = output if isinstance(output, tuple) else (output,)
    raw_outputs = deepcopy(outputs)

    # Once again, fold in token dimension into batch
    if is_lm and not use_residual_stream:
        inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(outputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        resid_stream_outputs = rearrange(raw_outputs[0], "b p d_hidden_combined -> (b p) d_hidden_combined")
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(torch.cat([x for x in outputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
    else:  # Inputs always tuple, and in this case we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
        outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    operator: Float[Tensor, "batch d_hidden_out"] = torch.div(outputs, inputs)
    _, d_hidden = operator.shape

    extended_replacement_idxs = torch.arange(d_hidden)
    extended_replacement_idxs = replacement_idxs
    operator[..., torch.arange(d_hidden)] = operator[..., extended_replacement_idxs]
    edited_output = operator * inputs

    if output_is_tuple:     # Put back in tuple form if ouput should have been tuple
        return resid_stream_outputs, edited_output

    return edited_output


def delete_cluster_duplicate_forward_hook_fn(
    module: torch.nn.Module,
    inputs: Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    output: Union[
        Float[Tensor, "batch d_hidden"],
        Float[Tensor, "batch pos d_hidden"],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ],
    hooked_data: dict[str, Any],
    hook_name: str,
    data_key: Union[str, list[str]],
    cluster_idxs: list[Int[Tensor, "d_hidden"]],
    use_residual_stream: bool,
) -> Union[
        tuple[Float[Tensor, "batch d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden"]],
        tuple[Float[Tensor, "batch pos d_hidden1"],
              Float[Tensor, "batch pos d_hidden2"]],
    ]:
    """Based on results showing functions within a cluster are copies of each other, delete all but
    one.

    Hook mlp_out layer with unembed matrix.

    Adjust weight matrices to reflect deletion.
    """
    output_is_tuple = True if isinstance(output, tuple) else False
    is_lm = True if inputs[0].dim() == 3 else False
    outputs = output if isinstance(output, tuple) else (output,)
    raw_outputs = deepcopy(outputs)

    if is_lm and not use_residual_stream:
        inputs = rearrange(inputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(outputs[1], "b p d_hidden_combined -> (b p) d_hidden_combined")
        resid_stream_outputs = rearrange(raw_outputs[0], "b p d_hidden_combined -> (b p) d_hidden_combined")
    elif is_lm and use_residual_stream:
        inputs = rearrange(torch.cat([x for x in inputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
        outputs = rearrange(torch.cat([x for x in outputs], dim=-1), "b p d_hidden_combined -> (b p) d_hidden_combined")
    else:  # Inputs always tuple, and in this case we don't have LM
        inputs = torch.cat([x.detach().clone() for x in inputs], dim=-1)
        outputs = torch.cat([x.detach().clone() for x in outputs], dim=-1)

    with torch.no_grad():
        for cluster_num, idxs in enumerate(cluster_idxs):
            idxs_to_delete = idxs[1:]
            inputs[..., idxs_to_delete] = 0 # Kill off all but first member of cluster
            w = module.weight.data
            # Add weight rows of deleted neurons to weight row of first member of cluster
            w[idxs[0]] += torch.sum(w[idxs_to_delete, :], dim=0)

        edited_output = inputs @ w.T

    if output_is_tuple:     # Put back in tuple form if ouput should have been tuple
        return resid_stream_outputs, edited_output

    return edited_output
