import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import yaml
from feature_visualization import parse_activation_data
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sae_vis.data_storing_fns import (
    FeatureData,
    FeatureVisParams,
    HistogramData,
    MiddlePlotsData,
    MultiFeatureData,
    SequenceMultiGroupData,
)
from sae_vis.utils_fns import QuantileCalculator, TopK, get_device, process_str_tok
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from rib.analysis_utils import get_rib_acts, get_rib_acts_and_resid_final
from rib.data import (
    BlockVectorDataset,
    BlockVectorDatasetConfig,
    DatasetConfig,
    HFDatasetConfig,
    ModularArithmeticDataset,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.data_accumulator import (
    Edges,
    collect_dataset_means,
    collect_gram_matrices,
    collect_interaction_edges,
)
from rib.distributed_utils import (
    DistributedInfo,
    adjust_logger_dist,
    get_device_mpi,
    get_dist_info,
)
from rib.hook_manager import HookedModel
from rib.interaction_algos import InteractionRotation, calculate_interaction_rotations
from rib.loader import load_dataset, load_model
from rib.log import logger
from rib.models import (
    MLPConfig,
    ModularMLPConfig,
    SequentialTransformer,
    SequentialTransformerConfig,
)
from rib.models.components import Unembed
from rib.rib_builder import RibBuildConfig, RibBuildResults
from rib.settings import REPO_ROOT
from rib.types import TORCH_DTYPES, IntegrationMethod, RootPath, StrDtype
from rib.utils import (
    check_out_file_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    get_chunk_indices,
    handle_overwrite_fail,
    load_config,
    replace_pydantic_model,
    set_seed,
)


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
        logger.info(f"Dataset length: {len(dataset)}")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.gram_batch_size,
        shuffle=False,
    )

    for batch in data_loader:
        print(torch.tensor(batch[0]).shape)

    interaction_rotations = rib_results.interaction_rotations

    model.eval()
    hooked_model = HookedModel(model)

    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)

    last_section_name = list(hooked_model.model.sections.keys())[-1]
    assert len(hooked_model.model.sections[last_section_name]) == 1
    last_section = hooked_model.model.sections[last_section_name][0]
    assert isinstance(last_section, Unembed)
    W_U = last_section.W_U
    # TODO correct transpose?
    rib_acts = get_rib_acts_and_resid_final(
        hooked_model, data_loader, interaction_rotations, device=device, dtype=dtype
    )
    # TensorDataset get data
    # tokens = torch.cat([x[0] for x in dataset], dim=0)
    # that doesn't respect n_ctx

    vocab_dict = {v: process_str_tok(k) for k, v in tokenizer.get_vocab().items()}

    resid_post = rib_acts.pop("unembed")

    def find_corresponding_index(selected_node_layer: str) -> int:
        for i, info in enumerate(interaction_rotations):
            if info.node_layer == selected_node_layer:
                return i
        raise ValueError(f"Could not find {selected_node_layer} in interaction_rotations")

    selected_node_layer = "ln1.3"
    corresponding_index = find_corresponding_index(selected_node_layer)

    rib_acts_mlp_in_3 = rib_acts[selected_node_layer]
    assert interaction_rotations[corresponding_index].node_layer == selected_node_layer
    C_mlp_in_3 = interaction_rotations[corresponding_index].C
    C_pinv_mlp_in_3 = interaction_rotations[corresponding_index].C_pinv

    # def process_vocab_dict(tokenizer: PreTrainedTokenizerBase) -> dict[int, str]:
    #     """
    #     Creates a vocab dict suitable for dashboards by replacing all the special tokens with their
    #     HTML representations. This function is adapted from sae_vis.create_vocab_dict()
    #     """
    #     vocab_dict: dict[str, int] = tokenizer.get_vocab()
    #     vocab_dict_processed: dict[int, str] = {v: process_str_tok(k) for k, v in vocab_dict.items()}
    #     return vocab_dict_processed
    # resid_post.shape = (n_samples, n_ctx, d_resid)

    fvp = FeatureVisParams(include_left_tables=False)

    list_of_batches = []
    for data, label in data_loader:
        list_of_batches.append(data)
    data = torch.cat(list_of_batches, dim=0)

    # From SAE VIZ demo.ipynb
    # tokens torch.Size([1024, 128]) = batch ctx
    # all_feat_acts torch.Size([1024, 128, 10]) = batch ctx features
    # final_resid_acts torch.Size([1024, 128, 512]) = batch ctx resid
    # feature_resid_dirs torch.Size([10, 512]) = features resid
    # feature_indices_list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = features
    # W_U torch.Size([512, 48262]) = resid vocab
    # vocab_dict {22519: '&nbsp;Along',

    # mfd = parse_activation_data(
    #     data,  # 50, 200 = n_samples, n_ctx
    #     rib_acts_mlp_in_3,  # 50 200 127 = n_samples, n_ctx, d_rib
    #     resid_post,  # 50 200 64 = n_samples, n_ctx, d_embed
    #     feature_resid_dirs=C_mlp_in_3.T,  # 127 130 = d_rib d_double
    #     # C or C_inv?
    #     feature_indices_list=[0, 1, 2, 3, 4, 5],
    #     W_U=torch.cat([W_U, W_U], dim=0),  # 130 50257 = 2*(d_embed+d_bias) d_vocab
    #     vocab_dict=vocab_dict,  # fine.
    #     fvp=fvp,
    # )

    #    new_resid_post = resid_post_group - resid_post_feature_effect

    # Main issue: What's up with Logit Lens
    # How does a 130d RIB affect 64d resid? And why is W_U 65d?
    # W_U[0, :] can be the bias for every token, so remove
    # Now W_U is 64 dim just like resid

    mfd = parse_activation_data(
        data.cpu(),  # 50, 200 = n_samples, n_ctx
        rib_acts_mlp_in_3.to(torch.float32).cpu(),  # 50 200 64 = n_samples, n_ctx, d_rib
        resid_post.to(torch.float32).cpu(),  # 50 200 64 = n_samples, n_ctx, d_embed
        feature_resid_dirs=C_pinv_mlp_in_3[:, 1:].cpu().to(torch.float32),  # 64 65 = d_rib d_double
        feature_indices_list=range(len(C_pinv_mlp_in_3)),
        W_U=W_U[1:].cpu().to(torch.float32),  # 64 50257 = d_embed d_vocab
        vocab_dict=vocab_dict,  # fine.
        fvp=fvp,
    )

    generate_dashboard_html_files(mfd, "./")


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
