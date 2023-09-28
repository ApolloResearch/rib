"""
<This file should do for lms what `experiments/mnist_orthog_ablation/run_orthog_ablations.py` does
for mnist.>

Run an lm while rotating to and from a (truncated) orthogonal basis.

The process is as follows:
    1. Load an LM trained on a dataset and a test set of sentences.
    2. Calculate gram matrices at each node layer.
    3. Calculate a rotation matrix at each node layer representing the operation of rotating to and
        from the partial eigenbasis of the gram matrix. The partial eigenbasis is equal to the
        entire eigenbasis with the zeroed out eigenvectors corresponding to the n smallest
        eigenvalues. The values of n follow the ablation schedule given below.
    4. Run the test set through the LM, applying the rotations at each node layer, and calculate
        the resulting accuracy.
    5. Repeat steps 3 and 4 for a range of values of n, plotting the resulting accuracies.

A node layer is positioned at the input of each module_name specified in the config. In this script,
we don't create a node layer at the output of the final module, as ablating nodes in this layer
is not useful.

Usage:
    python run_lm_orthog_ablations.py <path/to/yaml_config_file>

"""

import json
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import DataLoader

from rib.ablations import ablate_and_test
from rib.data_accumulator import collect_gram_matrices
from rib.hook_manager import HookedModel
from rib.linalg import eigendecompose
from rib.loader import (
    create_modular_arithmetic_data_loader,
    load_sequential_transformer,
)
from rib.log import logger
from rib.types import TORCH_DTYPES
from rib.utils import (
    calc_ablation_schedule,
    eval_model_accuracy,
    load_config,
    overwrite_output,
    set_seed,
)


class Config(BaseModel):
    exp_name: Optional[str] = Field(
        None,
        description="The name of the experiment. If None, don't write results to file.",
    )
    tlens_pretrained: Optional[Literal["gpt2"]] = Field(
        None, description="Pretrained transformer lens model."
    )
    tlens_model_path: Optional[Path] = Field(
        None, description="Path to saved transformer lens model."
    )
    dataset: Literal["modular_arithmetic", "wikitext"] = Field(
        ...,
        description="The dataset to use to build the graph.",
    )
    batch_size: int = Field(..., description="The batch size to use when building the graph.")

    last_pos_only: bool = Field(
        False,
        description="Whether the unembed module should only be applied to the last position.",
    )
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point at which we start ablating every individual vector. If None, always ablate every vector.",
    )
    dtype: str = Field(..., description="The dtype to use when building the graph.")
    node_layers: list[str] = Field(
        ..., description="Names of the node layers to build the graph with."
    )
    seed: int

    @field_validator("dtype")
    def dtype_validator(cls, v):
        assert v in TORCH_DTYPES, f"dtype must be one of {TORCH_DTYPES}"
        return v


def run_ablations(
    hooked_model: HookedModel,
    data_loader: DataLoader,
    config: Config,
    graph_module_names: list[str],
    ablate_every_vec_cutoff: Optional[int],
    dtype: torch.dtype,
    device: str,
) -> dict[str, dict[int, float]]:
    """Rotate to and from orthogonal basis and compare accuracies with and without ablation.

    Args:
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        config: The config.
        graph_module_names: The names of the modules we want to build the graph around.
        ablate_every_vec_cutoff: The point in the ablation schedule to start ablating every vector.
        dtype: The data type to use for model computations.
        device: The device to run the model on.


    Returns:
        A dictionary mapping node lyae to accuracy results.
    """

    gram_matrices = collect_gram_matrices(
        hooked_model=hooked_model,
        module_names=graph_module_names,
        data_loader=data_loader,
        dtype=dtype,
        device=device,
        collect_output_gram=True,
        hook_names=config.node_layers,
    )
    results: dict[str, dict[int, float]] = {}
    for hook_name, module_name in zip(config.node_layers, graph_module_names):
        _, eigenvecs = eigendecompose(gram_matrices[hook_name])

        ablation_schedule = calc_ablation_schedule(
            ablate_every_vec_cutoff=ablate_every_vec_cutoff,
            n_vecs=len(eigenvecs),
        )
        module_accuracies: dict[int, float] = ablate_and_test(
            hooked_model=hooked_model,
            module_name=module_name,
            interaction_rotation=eigenvecs,
            interaction_rotation_pinv=eigenvecs.T,
            test_loader=data_loader,
            device=device,
            ablation_schedule=ablation_schedule,
            hook_name=hook_name,
        )

        results[hook_name] = module_accuracies

    return results


def main(config_path_str: str) -> None:
    config_path = Path(config_path_str)
    config = load_config(config_path, config_model=Config)

    set_seed(config.seed)

    if config.exp_name is not None:
        out_file = Path(__file__).parent / "out" / f"{config.exp_name}_orthog_ablation_results.json"
        if out_file.exists() and not overwrite_output(out_file):
            print("Exiting.")
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)

    assert (
        config.tlens_pretrained is None and config.tlens_model_path is not None
    ), "Currently can't build graphs for pretrained models due to memory limits."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]
    seq_model, _ = load_sequential_transformer(
        node_layers=config.node_layers,
        last_pos_only=config.last_pos_only,
        tlens_pretrained=config.tlens_pretrained,
        tlens_model_path=config.tlens_model_path,
        dtype=dtype,
        device=device,
    )

    seq_model.eval()
    seq_model.to(device=torch.device(device), dtype=dtype)
    seq_model.fold_bias()
    hooked_model = HookedModel(seq_model)

    assert config.dataset == "modular_arithmetic", "Currently only supports modular arithmetic."

    data_loader = create_modular_arithmetic_data_loader(
        shuffle=True,
        return_set="test",
        tlens_model_path=config.tlens_model_path,
        batch_size=config.batch_size,
        seed=config.seed,
        frac_train=0.5,  # Take a random 50% split of the dataset
    )

    # Test model accuracy before ablation
    accuracy = eval_model_accuracy(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Accuracy before ablation: %.2f%%", accuracy * 100)

    graph_module_names = [f"sections.{sec}" for sec in seq_model.sections if sec != "pre"]

    accuracies: dict[str, dict[int, float]] = run_ablations(
        hooked_model=hooked_model,
        data_loader=data_loader,
        config=config,
        graph_module_names=graph_module_names,
        ablate_every_vec_cutoff=config.ablate_every_vec_cutoff,
        dtype=dtype,
        device=device,
    )
    results = {
        "exp_name": config.exp_name,
        "config": json.loads(config.model_dump_json()),
        "accuracies": accuracies,
    }
    if config.exp_name is not None:
        with open(out_file, "w") as f:
            json.dump(results, f)
        logger.info("Wrote results to %s", out_file)


if __name__ == "__main__":
    fire.Fire(main)
