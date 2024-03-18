from einops import einsum, rearrange
from jaxtyping import Float, Int
from sae_vis.data_fetching_fns import get_sequences_data
from sae_vis.data_storing_fns import (
    FeatureData,
    FeatureVisParams,
    HistogramData,
    MiddlePlotsData,
    MultiFeatureData,
    SequenceMultiGroupData,
)
from sae_vis.utils_fns import QuantileCalculator, TopK
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_activation_data(
    tokens: Int[Tensor, "batch n_ctx"],
    feature_acts: Float[Tensor, "batch n_ctx n_feats"],
    final_resid_acts: Float[Tensor, "batch n_ctx d_resid_final"],
    feature_resid_dirs: Float[Tensor, "n_feats d_resid_feat"],
    feature_indices_list: list[int],
    W_U: Float[Tensor, "d_resid_final d_vocab"],
    vocab: dict[int, str] | AutoTokenizer,
    fvp: FeatureVisParams,
) -> MultiFeatureData:
    """Convert generic activation data into a MultiFeatureData object, which can be used to create
    the feature-centric visualisation.

    final_resid_acts, feature_resid_dir, and W_U are used just for the logit lens.

    Not using left_tables_data, always sets left_tables_data attribute of FeatureData to None.

    Args:
        tokens: The inputs to the model
        feature_acts: The activations values of the features
        final_resid_acts: The activations of the final layer of the model
        feature_resid_dirs: The directions that each feature writes to the logit output
        feature_indices_list: The indices of the features we're interested in. Note that if you
            also need to adjust feature_resid_dirs to be only the features you're interested in.
        W_U: The weights of the logit lens
        vocab_dict: A dictionary mapping vocab indices to strings
        fvp: FeatureVisParams, containing a bunch of settings. See the FeatureVisParams docstring in
                sae_vis for more information.
    """
    sequence_data_dict: dict[int, SequenceMultiGroupData] = {}  # right hand side visualisation
    middle_plots_data_dict: dict[int, MiddlePlotsData] = {}  # middle visualisation
    feature_dashboard_data: dict[int, FeatureData] = {}

    # Calculate all data for the right-hand visualisations, i.e. the sequences
    for i, feat in tqdm(
        enumerate(feature_indices_list),
        desc="Getting sequence data",
        total=len(feature_indices_list),
    ):
        # Add this feature's sequence data to the list
        sequence_data_dict[feat] = get_sequences_data(
            tokens=tokens,
            feat_acts=feature_acts[..., i],
            resid_post=final_resid_acts,
            feature_resid_dir=feature_resid_dirs[i],
            W_U=W_U,
            fvp=fvp,
            dtype=feature_acts.dtype,
        )

    # Only calculate the middle visualisations if the architecture makes logit lens easy (we test
    # this by checking the shape -- in our case this implies that that layer allows for logit lens).
    logit_lens = final_resid_acts.shape[-1] == feature_resid_dirs.shape[-1]
    if logit_lens:
        # Get the logits of all features (i.e. the directions this feature writes to the logit output)
        logits = einsum(
            feature_resid_dirs,
            W_U,
            "feats d_model, d_model d_vocab -> feats d_vocab",
        )
        for i, (feat, logit) in enumerate(zip(feature_indices_list, logits, strict=True)):
            # Get data for logits (the histogram, and the table)
            logits_histogram_data = HistogramData(logit, n_bins=40, tickmode="5 ticks")
            top10_logits = TopK(logit, k=15, largest=True)
            bottom10_logits = TopK(logit, k=15, largest=False)

            # Get data for feature activations histogram (the title, and the histogram)
            feat_acts = feature_acts[..., i]
            nonzero_feat_acts = feat_acts[feat_acts > 0]
            frac_nonzero = nonzero_feat_acts.numel() / feat_acts.numel()
            freq_histogram_data = HistogramData(nonzero_feat_acts, n_bins=40, tickmode="ints")

            # Create a MiddlePlotsData object from this, and add it to the dict
            middle_plots_data_dict[feat] = MiddlePlotsData(
                bottom10_logits=bottom10_logits,
                top10_logits=top10_logits,
                logits_histogram_data=logits_histogram_data,
                freq_histogram_data=freq_histogram_data,
                frac_nonzero=frac_nonzero,
            )

    # Return the output, as a dict of FeatureData items
    for i, feat in enumerate(feature_indices_list):
        feature_dashboard_data[feat] = FeatureData(
            # Data-containing inputs (for the feature-centric visualisation)
            sequence_data=sequence_data_dict[feat],
            middle_plots_data=middle_plots_data_dict[feat] if logit_lens else None,
            # Non data-containing inputs
            left_tables_data=None,
            feature_idx=feat,
            vocab_dict=vocab,
            fvp=fvp,
        )

    # Also get the quantiles, which will be useful for the prompt-centric visualisation
    feature_act_quantiles = QuantileCalculator(
        data=rearrange(feature_acts, "... feats -> feats (...)")
    )
    return MultiFeatureData(feature_dashboard_data, feature_act_quantiles)
