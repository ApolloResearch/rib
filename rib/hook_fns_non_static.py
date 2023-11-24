"""Contains hook functions whose only purpose is to change the model's activations on forward pass,
not store data."""
from copy import deepcopy
from typing import Any, Union

import torch
import math
from einops import rearrange
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
    centroid_idxs: list[int],
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
        for cluster_num, (idxs, centroid_idx) in enumerate(zip(cluster_idxs, centroid_idxs)):
            assert centroid_idx in idxs, "centroid idx does not match cluster"
            idx_to_keep = torch.where(idxs == centroid_idx)[0]
            delete_mask = torch.arange(idxs.size(0)) != idx_to_keep
            for name, param in module.named_parameters():
                if "W" in name: # Unembed matrix
                    w = param # [input, output]

            # # Commented out because scale factor method here is very sensitive to small values so not as good as second one used below
            # for row_idx in idxs[1:]: # Find data-point agnostic activation function scale factor (see writeup)
            #     act_to_throw = inputs[:, row_idx]
            #     act_to_keep = inputs[:, idxs[0]]
            #     division_result = torch.div(act_to_throw, act_to_keep)
            #     # Do not include inf or nan in mean calculation
            #     scale_factor = torch.mean(division_result[torch.isfinite(division_result)])
            #     w[idxs[0], :] += torch.div(w[row_idx, :], scale_factor)

            for row_idx in idxs[delete_mask]:
                act_to_throw = inputs[:, row_idx].squeeze()
                act_to_keep = inputs[:, centroid_idx].squeeze()
                denominator = torch.dot(act_to_keep, act_to_keep)
                tol = 1e-10
                if torch.abs(denominator) < tol: continue
                scale_factor = torch.div(torch.dot(act_to_throw, act_to_keep), torch.dot(act_to_keep, act_to_keep))
                if torch.abs(scale_factor) < tol: continue
                w[centroid_idx, :] += torch.div(w[row_idx, :], scale_factor)

            inputs[:, idxs[delete_mask]] = 0 # Kill off all but first member of cluster

        edited_output = inputs @ w

    if output_is_tuple:     # Put back in tuple form if ouput should have been tuple
        return resid_stream_outputs, edited_output

    return edited_output
