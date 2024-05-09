import csv
import warnings
from pathlib import Path
from typing import Optional, Union

import einops
import fire
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.analysis_utils import get_rib_acts
from rib.data import ModularArithmeticDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import load_dataset, load_model
from rib.rib_builder import RibBuildResults
from rib.utils import (
    check_out_file_overwrite,
    handle_overwrite_fail,
    replace_pydantic_model,
)


def fft2(acts: torch.Tensor, use_numpy=False, dtype=torch.complex128) -> torch.Tensor:
    """Compute the 2D FFT of the activations over the first two dimensions, assumed to be x and y

    Args:
        acts (torch.Tensor): The activations to transform. If acts is of shape (x, y, ...), then the
            returned tensor will be of shape (x, y, ...) with x and y FFT'ed.

    Returns:
        acts (torch.Tensor): The FFT'ed activations.
    """
    assert acts.shape[0] == acts.shape[1], "acts must be square (same x and y dim)"
    if use_numpy:
        in_device = acts.device
        out_dtype = (acts + 1j).dtype
        acts = acts.clone().detach().numpy()
        return torch.tensor(np.fft.fft2(acts, axes=(0, 1)), device=in_device, dtype=out_dtype)
    else:
        out_dtype = (acts + 1j).dtype
        return torch.fft.fft2(acts.to(dtype), dim=(0, 1)).to(out_dtype)


def rfft2(acts: torch.Tensor, use_numpy=False, dtype=torch.complex128) -> torch.Tensor:
    """Compute the 2D RFFT of the activations over the first two dimensions, assumed to be x and y

    Args:
        acts (torch.Tensor): The activations to transform. If acts is of shape (x, y, ...), then the
            returned tensor will be of shape (x, y, ...) with x and y FFT'ed.

    Returns:
        acts (torch.Tensor): The FFT'ed activations. Shape is (n // 2 + 1, n)
    """
    assert acts.shape[0] == acts.shape[1], "acts must be square (same x and y dim)"
    if use_numpy:
        in_device = acts.device
        out_dtype = (acts + 1j).dtype
        acts = acts.clone().detach().numpy()
        return torch.tensor(np.fft.rfft2(acts, axes=(0, 1)), device=in_device, dtype=out_dtype)
    else:
        out_dtype = (acts + 1j).dtype
        center_index = acts.shape[0] // 2 + 1
        return torch.fft.fft2(acts.to(dtype), dim=(0, 1)).to(out_dtype)[..., :center_index, :]


def to_real(acts: torch.Tensor, rtol=1e-6, atol=1e-6, eps=1e-10) -> torch.Tensor:
    """Make sure no imaginary part is present in the activations."""

    ratio = acts.imag.abs() / (acts.real.abs() + eps)
    if not torch.isclose(ratio, torch.zeros_like(ratio), rtol=rtol, atol=atol).all():
        warnings.warn(f"Ignored imaginary part {ratio.argmax()} ({acts[ratio.argmax()]:.2e})")

    return acts.real


def compute_label_list(
    acts: Float[Tensor, "p p n"],
    max_dimension=10,
    norm_scale: float = 1,
    p=113,
    rtol=1e-3,
    atol=1e-7,
) -> list:
    """For each direction (dimension), compute the labels that account for most of the variance.

    Args:
        acts (Float[Tensor, "p p n"]): Activations of shape (p, p, n).
        max_dimension (int, optional): Maximum dimension to compute labels for. Defaults to 10.
        norm_scale (float, optional): Divide the norms of the labels by this factor. Used to adjust
            norms in separatelty-computed embedding activations. Defaults to 1.
        p (int, optional): Modulus aka size of the activations. Defaults to 113.
        rtol (float, optional): Relative tolerance for torch.testing.assert_close. Defaults to 1e-3.
        atol (float, optional): Absolute tolerance for torch.testing.assert_close. Defaults to 1e-7.

    """
    label_list = []
    px, py, dimensions = acts.shape
    assert p == px == py, "Only works for square activations ofn size p"
    assert p % 2 == 1, "Only works for odd p"
    fft_acts = fft2(acts)
    size = lambda x: (x.abs() ** 2).sum()
    for dimension in tqdm(range(min(max_dimension, dimensions)), desc="Iterating over dimensions"):
        labels = {}
        a = fft_acts[:, :, dimension]
        # assert torch.allclose(size(a), size(ar), rtol=rtol, atol=atol)
        total = size(a) * norm_scale
        # Constant term contribution
        labels["const"] = size(a[0, 0]) / total
        # x and y term contributions
        for i in range(1, p // 2 + 1):
            cosx = (size(a[i, 0]) + size(a[p - i, 0])) / total
            cosy = (size(a[0, i]) + size(a[0, p - i])) / total
            if abs(cosx - cosy) < 0.01:
                labels[f"cos({i}x) + cos({i}y)"] = cosx + cosy
            else:
                labels[f"cos({i}x)"] = cosx
                labels[f"cos({i}y)"] = cosy
        # x y crossterms.
        # First compute "plus plus" and "plus minus" terms to account for symmetry
        acts_shifted = torch.fft.fftshift(a, dim=[0, 1])
        A_pp = acts_shifted[p // 2 + 1 :, p // 2 + 1 :].abs()
        A_pm = acts_shifted.flip(dims=(1,))[p // 2 + 1 :, p // 2 + 1 :].abs()
        # Compute both cos(x+y) cos(x-y) and cos(x)cos(y) sin(x)sin(y) bases
        a_plus = np.sqrt(2) * A_pp
        a_minus = np.sqrt(2) * A_pm
        a_coscos = 1 * (A_pp + A_pm)
        a_sinsin = 1 * (A_pp - A_pm)
        # Check normalization
        norm_1 = (size(a_plus) + size(a_minus)) / size(a[1:, 1:])
        if (norm_1 - 1).abs() > rtol and size(a[1:, 1:]) / total > rtol:
            warnings.warn(f"Normalization of plus/minus != 1: {norm_1}")
        norm_2 = (size(a_coscos) + size(a_sinsin)) / size(a[1:, 1:])
        if (norm_2 - 1).abs() > rtol and size(a[1:, 1:]) / total > rtol:
            warnings.warn(f"Normalization of coscos/sinsin != 1: {norm_2}")
        # Check whether to use cos(w_x x + w_y y), cos(w_x x - w_y y) or
        # cos(w_x x)*cos(w_y y) + sin(w_x x)*sin(w_y y) convention. We usually expect
        # that in one of the conventions, terms will me mostly of one kind, i.e. either
        # * cos(w_x x + w_y y) >> cos(w_x x - w_y y) or
        # * cos(w_x x)*cos(w_y y) >> sin(w_x x)*sin(w_y y)
        # Chose the convention where we get the >> condition for nicer labels.
        if size(a_plus) / size(a_minus) > size(a_coscos) / size(a_sinsin):
            # Use cos(w_x x + w_y y), cos(w_x x - w_y y) convention
            for i in range(p // 2):
                for j in range(p // 2):
                    freqs = torch.fft.fftfreq(p)
                    freqs_shifted = torch.fft.fftshift(freqs, dim=0)
                    fx = round(p * freqs_shifted[i + p // 2 + 1].item(), 0)  # == i
                    fy = round(p * freqs_shifted[j + p // 2 + 1].item(), 0)  # == j
                    assert fx == float(i + 1) and fy == float(
                        j + 1
                    ), f"FFT frequencies are off, got {fx} and {fy}"
                    labels[f"cos({i+1}x + {j+1}y)"] = size(a_plus[i, j]) / total
                    labels[f"cos({i+1}x - {j+1}y)"] = size(a_minus[i, j]) / total
        else:
            # Use cos(w_x x)*cos(w_y y) + sin(w_x x)*sin(w_y y) convention
            for i in range(p // 2):
                for j in range(p // 2):
                    freqs = torch.fft.fftfreq(p)
                    freqs_shifted = torch.fft.fftshift(freqs, dim=0)
                    fx = round(p * freqs_shifted[i + p // 2 + 1].item(), 0)
                    fy = round(p * freqs_shifted[j + p // 2 + 1].item(), 0)
                    assert fx == float(i + 1) and fy == float(
                        j + 1
                    ), f"FFT frequencies are off, got {fx} and {fy} for {i} and {j}"
                    labels[f"cos({i+1}x) * cos({j+1}y)"] = size(a_coscos[i, j]) / total
                    labels[f"sin({i+1}x) * sin({j+1}y)"] = size(a_sinsin[i, j]) / total
        # Check that the sum of labels is 1
        sum_of_labels = sum(labels.values())
        assert (sum_of_labels - 1 / norm_scale).abs() < 1e-3, f"Sum of labels is {sum_of_labels}"
        label_list.append(labels)
    return label_list


def filter_label_list(label_list, topp=0.995, maxlen=7):
    label_list = label_list.copy()
    """ For each dimension, filter the labels so that only those that account for
    topp of the magnitude are kept. Also make it so that n_labels <= maxlen.
    """
    filtered_label_list = []
    for labels in label_list:
        filtered_labels = {}
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
        total = sum([x[1] for x in sorted_labels])
        # Assert total=1
        assert (total - 1).abs() < 1e-3, f"Total is {total}"
        cumsum = 0
        for label, magnitude in sorted_labels:
            cumsum += magnitude
            filtered_labels[label] = magnitude
            if cumsum / total > topp:
                break
            if magnitude / total < 0.005:
                break
        # Make sure we have at most maxlen labels for readability
        filtered_labels = dict(list(filtered_labels.items())[0:maxlen])
        filtered_label_list.append(filtered_labels)
    return filtered_label_list


def main(
    interaction_graph_path: Union[str, Path],
    max_dim: int = 10,
    out_file: Optional[str] = None,
    force: bool = False,
):
    """Generate mod_add labels for the interaction graph.

    Args:
        interaction_graph_path (Union[str, Path]): Path to the interaction graph.
    """

    print("Loading", interaction_graph_path)

    results = RibBuildResults(**torch.load(interaction_graph_path))

    if out_file is not None:
        if not check_out_file_overwrite(Path(out_file), force):
            handle_overwrite_fail()

    assert results.config.dataset is not None  # for mypy
    dataset_config = replace_pydantic_model(
        results.config.dataset, {"return_set": "all", "fn_name": "add"}
    )
    assert isinstance(dataset_config, ModularArithmeticDatasetConfig)

    model = load_model(results.config, device="cuda", dtype=torch.float32)
    dataset = load_dataset(dataset_config)

    dataloader = DataLoader(dataset, batch_size=99999, shuffle=False)
    hooked_model = HookedModel(model)

    non_output_interactions = (
        results.interaction_rotations[:-1]
        if results.interaction_rotations[-1].node_layer == "output"
        else results.interaction_rotations
    )
    acts = get_rib_acts(
        hooked_model,
        data_loader=dataloader,
        interaction_rotations=non_output_interactions,
    )

    embedding = acts["ln1.0"].view(113, 113, 3, -1)
    embedding_x, embedding_y, embedding_z = einops.rearrange(embedding, "x y p h -> p x y h")
    resid_mid = acts["ln2.0"].view(113, 113, -1)
    pre_unembed = acts["unembed"].view(113, 113, -1)

    full_label_str_list = []
    for name, act in zip(
        ["embedding", "resid_mid", "pre_unembed"],
        [
            # Can treat embedding separately by passing them without adding but adding them is a
            # handy trick to simplify the code. If you change that see embedding_treatment_separate
            # below.
            embedding_x + embedding_y + embedding_z,
            resid_mid,
            pre_unembed,
        ],
    ):
        print(f"Computing labels for {name}")
        embedding_treatment_separate = False
        if embedding_treatment_separate and name == "embedding":
            label_list_x = compute_label_list(act[0], max_dimension=max_dim, norm_scale=2)
            label_list_y = compute_label_list(act[1], max_dimension=max_dim, norm_scale=2)
            # Merge list of dicts, renaming keys by appending a 0 or 1
            label_list = []
            for labels_x, labels_y in zip(label_list_x, label_list_y):
                labels = {}
                for label, magnitude in labels_x.items():
                    labels[label + "_0"] = magnitude
                for label, magnitude in labels_y.items():
                    labels[label + "_1"] = magnitude
                label_list.append(labels)
        else:
            label_list = compute_label_list(act, max_dimension=max_dim)
        filtered_label_list = filter_label_list(label_list)
        label_str_list = []
        for dimension, labels in enumerate(filtered_label_list):
            label_str = "|".join(
                f"{magnitude*100:.0f}% {label}" for label, magnitude in labels.items()
            )
            label_str_list.append(label_str)
        full_label_str_list.append(label_str_list)
    # Output layer
    output_str_list = [f"output_{out}" for out in range(114)]
    full_label_str_list.append(output_str_list)
    # Save to file
    if out_file is not None:
        with open(out_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(full_label_str_list)
        print(f"Saved labels to {out_file}")
    return full_label_str_list


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: "")
