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
from rib.data import DatasetConfig, HFDatasetConfig
from rib.hook_manager import HookedModel
from rib.loader import load_dataset, load_model
from rib.log import logger
from rib.models import SequentialTransformer
from rib.models.components import Unembed
from rib.rib_builder import RibBuildResults
from rib.types import TORCH_DTYPES


def generate_dashboard_html_files(multi_feature_data: MultiFeatureData, html_dir: str = ""):
    """Generates viewable HTML dashboards from the compressed multi_feature_dashboard_data"""
    for feature_idx in multi_feature_data.keys():
        filepath = html_dir + f"dashboard_feature-{feature_idx}.html"
        html_str = multi_feature_data[feature_idx].get_html()
        with open(filepath, "w") as f:
            f.write(html_str)


def main(
    results: RibBuildResults | str | Path,
    dataset_config: Optional[DatasetConfig] = None,
):
    if isinstance(results, (str, Path)):
        rib_results = RibBuildResults(**torch.load(results))
    else:
        rib_results = results

    config = rib_results.config
    device = "cuda"
    dtype = TORCH_DTYPES[config.dtype]
    model = load_model(config, device=device, dtype=dtype)

    if dataset_config is None:
        assert config.dataset is not None
        dataset_config = config.dataset = config.dataset
        dataset = load_dataset(
            dataset_config=config.dataset,
            model_n_ctx=model.cfg.n_ctx if isinstance(model, SequentialTransformer) else None,
            tlens_model_path=config.tlens_model_path,
        )
        logger.info(f"Dataset length: {len(dataset)}")  # type: ignore
    else:
        dataset = load_dataset(
            dataset_config=dataset_config, model_n_ctx=511, tlens_model_path=config.tlens_model_path
        )
        logger.info(f"Dataset length: {len(dataset)}")  # type: ignore
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.gram_batch_size,
        shuffle=False,
    )

    interaction_rotations = rib_results.interaction_rotations

    model.eval()
    hooked_model = HookedModel(model)

    assert isinstance(dataset_config, HFDatasetConfig)
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)

    last_section_name = list(hooked_model.model.sections.keys())[-1]
    assert len(hooked_model.model.sections[last_section_name]) == 1
    last_section = hooked_model.model.sections[last_section_name][0]
    assert isinstance(last_section, Unembed)
    W_U = last_section.W_U

    rib_acts = get_rib_acts_and_resid_final(
        hooked_model, data_loader, interaction_rotations, device=device, dtype=dtype
    )
    vocab_dict = {v: process_str_tok(k) for k, v in tokenizer.get_vocab().items()}

    resid_post = rib_acts.pop("unembed")

    def find_corresponding_index(selected_node_layer: str) -> int:
        for i, info in enumerate(interaction_rotations):
            if info.node_layer == selected_node_layer:
                return i
        raise ValueError(f"Could not find {selected_node_layer} in interaction_rotations")

    selected_node_layer = "ln1.7"
    corresponding_index = find_corresponding_index(selected_node_layer)

    rib_acts_mlp_in_3 = rib_acts[selected_node_layer]
    assert interaction_rotations[corresponding_index].node_layer == selected_node_layer
    C_mlp_in_3 = interaction_rotations[corresponding_index].C
    C_pinv_mlp_in_3 = interaction_rotations[corresponding_index].C_pinv

    list_of_batches = []
    for data, label in data_loader:
        list_of_batches.append(data)
    data = torch.cat(list_of_batches, dim=0)

    # TODO Can we get a viz where our RIB dimensions are weird (2xresid etc.)
    # TODO What's up with the FIXME below
    # TODO Please make device and dtype better
    # TODO Get a progress bar for parse_activation_data
    # Shape comparison from SAE VIZ demo.ipynb
    # tokens torch.Size([1024, 128]) = batch ctx
    # all_feat_acts torch.Size([1024, 128, 10]) = batch ctx features
    # final_resid_acts torch.Size([1024, 128, 512]) = batch ctx resid
    # feature_resid_dirs torch.Size([10, 512]) = features resid
    # feature_indices_list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = features
    # W_U torch.Size([512, 48262]) = resid vocab
    # vocab_dict {22519: '&nbsp;Along', ...

    fvp = FeatureVisParams(include_left_tables=False)

    assert C_mlp_in_3 is not None
    assert C_pinv_mlp_in_3 is not None
    assert resid_post is not None
    mfd = parse_activation_data(
        data.cpu(),  # torch.Size([500, 200])
        rib_acts_mlp_in_3.to(torch.float32).cpu(),  # torch.Size([500, 200, 64])
        resid_post.to(torch.float32).cpu(),  # torch.Size([500, 200, 64])
        feature_resid_dirs=C_pinv_mlp_in_3[:, 1:].cpu().to(torch.float32),  # torch.Size([64, 65])
        # (torch.Size([64, 64]) after slicing)
        feature_indices_list=range(len(C_pinv_mlp_in_3)),  # range(64)
        # #FIXME This breaks if feature_indices_list isn't the full range
        # ValueError: zip() argument 2 is longer than argument 1
        W_U=W_U[1:].cpu().to(torch.float32),  # torch.Size([65, 50257])
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

main(results_path, dataset_config=dataset_config)
