import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from feature_visualization import parse_activation_data
from sae_vis.data_storing_fns import FeatureVisParams, MultiFeatureData
from sae_vis.utils_fns import process_str_tok
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from rib.analysis_utils import get_rib_acts_and_resid_final
from rib.data import HFDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import load_dataset, load_model
from rib.log import logger
from rib.models import SequentialTransformer
from rib.models.components import Unembed
from rib.rib_builder import RibBuildResults
from rib.types import TORCH_DTYPES, StrDtype


def generate_dashboard_html_files(multi_feature_data: MultiFeatureData, html_dir: str = ""):
    """Generates viewable HTML dashboards from the compressed multi_feature_dashboard_data"""
    for feature_idx in multi_feature_data.keys():
        filepath = html_dir + f"dashboard_feature-{feature_idx}.html"
        html_str = multi_feature_data[feature_idx].get_html()
        with open(filepath, "w") as f:
            f.write(html_str)


def main(
    results: RibBuildResults | str | Path,
    dataset_cfg: Optional[HFDatasetConfig | str | Path] = None,
    device: str = "cuda",
    dtype_str: Optional[StrDtype] = None,
    batch_size: Optional[int] = None,
):
    """Generates a dashboard for the RIB activations and residuals of a model.

    Args:
        results: The results of a RIB build, or the path to a file containing the results.
        dataset_config: The config of the dataset used to build the RIB. If None, the dataset config
            from the RIB results will be used.
    """
    # Load results Cs
    if isinstance(results, (str, Path)):
        rib_results = RibBuildResults(**torch.load(results))
    elif isinstance(results, RibBuildResults):
        rib_results = results
    else:
        raise ValueError(f"results must be a path to a file or a RibBuildResults object")
    # Load dataset
    if dataset_cfg is None:
        assert rib_results.config.dataset is not None
        dataset_config = rib_results.config.dataset
        assert isinstance(dataset_config, HFDatasetConfig), "Only HFDatasetConfig is supported"
    elif isinstance(dataset_cfg, (str, Path)):
        dataset_config = HFDatasetConfig(**yaml.safe_load(open(dataset_cfg)))
    elif isinstance(dataset_cfg, HFDatasetConfig):
        dataset_config = dataset_cfg
    else:
        raise ValueError(f"dataset must be a path to a file, a DatasetConfig object, or None")
    # Set device, dtype, and batch_size
    rib_config = rib_results.config
    dtype = TORCH_DTYPES[dtype_str or rib_config.dtype]
    batch_size = batch_size or rib_config.gram_batch_size or rib_config.batch_size
    # Load model
    model = load_model(rib_config, device=device, dtype=dtype)
    model.eval()
    hooked_model = HookedModel(model)
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
    vocab_dict = {v: process_str_tok(k) for k, v in tokenizer.get_vocab().items()}
    # Load dataset
    dataset = load_dataset(
        dataset_config=dataset_config,
        model_n_ctx=model.cfg.n_ctx if isinstance(model, SequentialTransformer) else None,
        tlens_model_path=rib_config.tlens_model_path,
    )
    logger.info(f"Dataset length: {len(dataset)}")  # type: ignore
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    data = torch.cat([data for data, label in data_loader], dim=0)

    # Get W_U -- this code assumes that Unembed is its own section i.e. that "unembed" is in the
    # list of node layers. We test this assumption below.
    last_section_name = list(hooked_model.model.sections.keys())[-1]
    assert len(hooked_model.model.sections[last_section_name]) == 1
    last_section = hooked_model.model.sections[last_section_name][0]
    assert isinstance(last_section, Unembed)
    W_U = last_section.W_U

    # Get Cs & RIB activations over the dataset
    interaction_rotations = rib_results.interaction_rotations
    acts = get_rib_acts_and_resid_final(
        hooked_model, data_loader, interaction_rotations, device=device, dtype=dtype
    )
    resid_post = acts.pop("unembed")

    # Select a layer for the visualization -- could do this for all layers
    def find_corresponding_index(selected_node_layer: str) -> int:
        for i, info in enumerate(interaction_rotations):
            if info.node_layer == selected_node_layer:
                return i
        raise ValueError(f"Could not find {selected_node_layer} in interaction_rotations")

    selected_node_layer = "ln1.7"
    corresponding_index = find_corresponding_index(selected_node_layer)

    # Get acts and C for the selected layer
    rib_acts = acts[selected_node_layer]
    C = interaction_rotations[corresponding_index].C
    C_pinv = interaction_rotations[corresponding_index].C_pinv
    assert C is not None, "Selected layer does not have a C"
    assert C_pinv is not None, "Selected layer does not have a C_pinv [impossible, has C?]"

    # TODO Can we get a viz where our RIB dimensions are weird (2xresid etc.)
    # Smaller 0
    # Shape comparison from SAE VIZ demo.ipynb
    # tokens torch.Size([1024, 128]) = batch ctx
    # all_feat_acts torch.Size([1024, 128, 10]) = batch ctx features
    # final_resid_acts torch.Size([1024, 128, 512]) = batch ctx resid
    # feature_resid_dirs torch.Size([10, 512]) = features resid
    # feature_indices_list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = features
    # W_U torch.Size([512, 48262]) = resid vocab
    # vocab_dict {22519: '&nbsp;Along', ...

    fvp = FeatureVisParams(include_left_tables=False)

    assert resid_post is not None
    # device: Note that sae_viz uses their get_device function which always uses CUDA if available
    mfd = parse_activation_data(
        data.to(device),  # torch.Size([500, 200])
        rib_acts.to(dtype).to(device),  # torch.Size([500, 200, 64])
        resid_post.to(dtype).to(device),  # torch.Size([500, 200, 64])
        feature_resid_dirs=C_pinv[:, 1:].to(device).to(dtype),  # torch.Size([64, 65])
        # (torch.Size([64, 64]) after slicing)
        feature_indices_list=range(len(C_pinv)),  # range(64)
        # ValueError: zip() argument 2 is longer than argument 1
        W_U=W_U[1:].to(device).to(dtype),  # torch.Size([65, 50257])
        # (torch.Size([64, 50257]) after slicing
        vocab_dict=vocab_dict,  # 50257 entries
        fvp=fvp,
    )
    os.makedirs("./html/", exist_ok=True)
    generate_dashboard_html_files(mfd, "./html/")


results_path = "/mnt/ssd-interp/stefan/large_rib_runs/tinystories_scaling/stored_rib_stochpos1_ctx200_alllayers_10M/tinystories_nnib_samples50000_ctx200_rib_Cs.pt"

dataset_config = HFDatasetConfig(
    **yaml.safe_load(
        """
dataset_type: huggingface
name: roneneldan/TinyStories # or skeskinen/TinyStories-GPT4, but not clear if part of training
tokenizer_name: EleutherAI/gpt-neo-125M
return_set: train
return_set_frac: null
n_documents: 1000  # avg ~235 toks / document
n_samples: 500
return_set_portion: first
n_ctx: 200 # needs to be <= 511 for the model to behave reasonably
"""
    )
)

main(results_path, dataset_cfg=dataset_config)
