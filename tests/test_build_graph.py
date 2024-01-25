"""Test various properties of an lm and mnist RIB graph.

Properties tested:
1. The size of the sum of activations in the interaction basis (calculated with
    (C.T @ gram_matrices[module_name] @ C).diag()) is equal to both the outgoing edges of a node
    (calculated E_hats[i].sum(0).abs()), and the absolute sorted Lambdas of that layer.
2. The output rotation (C) is an identity matrix (i.e. it should be the eigenspace)

Note that, when comparing tensors, we normalize by their max value to account for tensors of
various scales. This is because combining atol and rtol does not work particularly well for tensors
that have a small set of large numbers and a large set of small numbers.
"""

import einops
import pytest
import torch
from fancy_einsum import einsum
from torch.testing import assert_close
from torch.utils.data import DataLoader

from rib.analysis_utils import get_rib_acts, rotation_list_to_dict
from rib.data_accumulator import collect_dataset_means
from rib.hook_fns import acts_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.loader import load_model_and_dataset_from_rib_config
from rib.log import logger
from rib.models import SequentialTransformer
from rib.rib_builder import RibBuildConfig, RibBuildResults, rib_build
from rib.types import TORCH_DTYPES
from rib.utils import replace_pydantic_model
from tests.utils import (
    assert_is_close,
    assert_is_zeros,
    get_mnist_config,
    get_modular_arithmetic_config,
    get_modular_mlp_config,
    get_pythia_config,
)


def graph_build_test(config: RibBuildConfig, atol: float):
    results = rib_build(config)

    grams = results.gram_matrices
    Cs = results.interaction_rotations
    edges = results.edges

    # The output interaction matrix should be None if rotate_final_node_layer is False
    if not config.rotate_final_node_layer:
        assert (
            Cs[-1].C is None
        ), "The output interaction matrix should be None if rotate_final_node_layer is False"

    # We don't have edges or lambdas for the final layer in node_layers
    comparison_layers = config.node_layers[:-1]
    for i, module_name in enumerate(comparison_layers):
        # Get the module names from the grams
        act_size = (Cs[i].C.T @ grams[module_name] @ Cs[i].C).diag()
        if config.calculate_edges:
            if config.edge_formula == "squared":
                assert (edges[i].E_hat >= 0).all(), f"edges not >= 0 for {module_name}"
            assert (edges[i].E_hat != 0).any(), f"edges all zero for {module_name}"
            if config.edge_formula == "functional" and config.basis_formula == "(1-alpha)^2":
                # Check that the size of the sum of activations in the interaction basis is equal
                # to the outgoing edges of a node. The relation should hold only in this one config
                # case.
                edge_size = edges[i].E_hat.sum(0).abs()
                assert (
                    act_size.shape == edge_size.shape
                ), f"act_size and edge_size not same shape for {module_name}"
                assert_close(
                    act_size / act_size.abs().max(),
                    edge_size / edge_size.abs().max(),
                    rtol=0,
                    atol=atol,
                )

        if config.basis_formula not in [
            "svd",
            "neuron",
            "jacobian",
        ]:  # We don't have Lambdas for these / Lambdas for jacobian basis don't fulfill this
            # Check that the Lambdas are also the same as the act_size and edge_size
            # Note that the Lambdas need to be truncated to edge_size/act_size (this happens in
            # `rib.interaction_algos.build_sort_lambda_matrix)
            Lambda_raw = results.interaction_rotations[i].Lambda
            Lambda = torch.sort(Lambda_raw, descending=True).values[: len(act_size)]
            assert torch.allclose(
                act_size / act_size.abs().max(), Lambda / Lambda.max(), atol=atol
            ), f"act_size not equal to Lambdas for {module_name}"

    return results


def get_rib_acts_test(results: RibBuildResults, atol: float, batch_size=16):
    """Takes the results of a graph build and checks get_rib_acts computes the correct values.

    This requires:
    * loading the model and dataset
    * 1) calling get_rib_acts, which hooks the model and computes the rib acts from that
    * 2) using run_with_cache to get the output of the previous module and rotating with C
    * comparing the results of 1) and 2)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[results.config.dtype]
    model, dataset = load_model_and_dataset_from_rib_config(
        rib_config=results.config, device=device, dtype=dtype
    )
    model.to(device=torch.device(device), dtype=dtype)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    rib_acts = get_rib_acts(
        hooked_model=hooked_model,
        data_loader=data_loader,
        interaction_rotations=results.interaction_rotations,
        device=device,
        dtype=dtype,
    )
    for m_name, acts in rib_acts.items():
        assert acts.shape[0] == len(dataset), f"acts.shape[0] != len(dataset) for {m_name}"

    # we choose a module to compare rib acts on and find the module immediately before it
    if isinstance(model, SequentialTransformer):
        # we choose the first module of node_layers, meaning the previous module is the last one
        # in sections.pre
        module_to_test = results.config.node_layers[0]
        prev_module_id = f"sections.pre.{len(model.sections['pre']) - 1}"
    else:
        # we arbitrarily choose the first layer.
        assert "layers.1" in results.config.node_layers, results.config.node_layers
        module_to_test = "layers.1"
        prev_module_id = "layers.0"

    prev_module_outputs = []
    with torch.inference_mode():
        hook = Hook(
            name=prev_module_id,
            data_key="acts",
            fn=acts_forward_hook_fn,
            module_name=prev_module_id,
        )
        for input, _ in data_loader:
            if input.dtype not in [torch.int32, torch.int64]:
                input = input.to(dtype=dtype)
            hooked_model.forward(input.to(device=device), hooks=[hook])
            cache_out = hooked_model.hooked_data[prev_module_id]["acts"]
            hooked_model.clear_hooked_data()
            prev_module_outputs.append(torch.concatenate(cache_out, dim=-1).cpu())
    prev_module_outputs = torch.concatenate(prev_module_outputs, dim=0)
    C = rotation_list_to_dict(results.interaction_rotations)[module_to_test].C
    test_rib_acts = einsum("... emb, emb rib -> ... rib", prev_module_outputs, C)
    utils_rib_acts = rib_acts[module_to_test].cpu()
    assert_is_close(utils_rib_acts, test_rib_acts, atol=atol, rtol=1e-5)
    return rib_acts


def get_means(results: RibBuildResults, atol: float, batch_size=16):
    """Takes the results of a graph build and runs collect_dataset_means."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[results.config.dtype]
    model, dataset = load_model_and_dataset_from_rib_config(
        rib_config=results.config, device=device, dtype=dtype
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    module_ids = [m_name for m_name in results.config.node_layers if m_name != "output"]
    if hasattr(model, "sections"):
        module_names = [model.module_id_to_section_id[m_name] for m_name in module_ids]
    else:
        module_names = module_ids
    return collect_dataset_means(
        hooked_model=hooked_model,
        module_names=module_names,
        data_loader=data_loader,
        device=device,
        dtype=dtype,
        collect_output_dataset_means=False,
        hook_names=module_ids,
    )


# @pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
        ("jacobian", "squared"),
    ],
)
def test_modular_arithmetic_build_graph(basis_formula, edge_formula):
    atol = 1e-12  # Works with 1e-7 for float32 and 1e-12 for float64. NEED 1e-5 for CPU
    config = get_modular_arithmetic_config(
        {"basis_formula": basis_formula, "edge_formula": edge_formula}
    )
    results = graph_build_test(config=config, atol=atol)
    get_rib_acts_test(results, atol=0)  # Need atol=1e-3 if float32


@pytest.mark.slow
def test_pythia_14m_build_graph():
    atol = 0  # Works with 1e-7 for float32 and 0 for float64
    config = get_pythia_config()
    results = graph_build_test(config=config, atol=atol)
    get_rib_acts_test(results, atol=0)


@pytest.mark.slow
def test_pythia_14m_build_graph_jacobian():
    atol = 0  # Works with 0 for batch_size 900 but not 1800
    updates = [
        # Runs in around 30s on a5000
        {"basis_formula": "jacobian"},
        {"dataset": {"return_set_n_samples": 1}},
        {"dataset": {"n_ctx": 2}},
        {"batch_size": 900},
        {"node_layers": ["ln2.1", "mlp_out.5", "unembed"]},
        {"calculate_edges": True},
        {"edge_formula": "stochastic"},
        {"n_stochastic_sources_edges": 1},
    ]
    config = get_pythia_config(*updates)
    results = graph_build_test(config=config, atol=atol)
    get_rib_acts_test(results, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
        ("jacobian", "squared"),
    ],
)
def test_mnist_build_graph(basis_formula, edge_formula):
    dtype_str = "float32"
    # Works with 1e-5 for float32 and 1e-15 (and maybe smaller) for float64.
    atol = 1e-5
    config = get_mnist_config(
        {"basis_formula": basis_formula, "edge_formula": edge_formula, "dtype": dtype_str}
    )
    results = graph_build_test(config=config, atol=atol)
    get_rib_acts_test(results, atol=atol)


@pytest.mark.parametrize(
    "basis_formula, edge_formula, dtype_str",
    [
        ("(1-alpha)^2", "squared", "float32"),
        ("(1-alpha)^2", "squared", "float64"),
        ("(1-0)*alpha", "squared", "float32"),
        ("(1-0)*alpha", "squared", "float64"),
        ("(1-alpha)^2", "functional", "float32"),
        ("(1-alpha)^2", "functional", "float64"),
        ("(1-0)*alpha", "functional", "float32"),
        ("(1-0)*alpha", "functional", "float64"),
        ("svd", "squared", "float32"),
        ("svd", "squared", "float64"),
        ("neuron", "squared", "float32"),
        ("neuron", "squared", "float64"),
        ("svd", "functional", "float32"),
        ("svd", "functional", "float64"),
        ("neuron", "functional", "float32"),
        ("neuron", "functional", "float64"),
    ],
)
def test_modular_mlp_build_graph(basis_formula, edge_formula, dtype_str, atol=1e-6):
    config = get_modular_mlp_config(
        {"basis_formula": basis_formula, "edge_formula": edge_formula, "dtype": dtype_str}
    )
    graph_build_test(config=config, atol=atol)


def rotate_final_layer_invariance(
    config_not_rotated: RibBuildConfig, rtol: float = 1e-7, atol: float = 0
):
    assert config_not_rotated.rotate_final_node_layer is False
    config_rotated = replace_pydantic_model(config_not_rotated, {"rotate_final_node_layer": True})
    edges_rotated = rib_build(config_rotated).edges
    edges_not_rotated = rib_build(config_not_rotated).edges

    # -1 has no edges, -2 is the final layer and changes
    comparison_layers = config_rotated.node_layers[:-2]
    for i, module_name in enumerate(comparison_layers):
        # E_hats[i] is a tuple (name, tensor)
        logger.info(("Comparing", module_name))
        rot = edges_rotated[i][1]
        notrot = edges_not_rotated[i][1]
        # Check shape
        assert (
            rot.shape == notrot.shape
        ), f"edges_not_rotated and edges_rotated not same shape for {module_name}"
        # Check values
        assert_close(
            rot,
            notrot,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.skip
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
        ("jacobian", "squared"),
    ],
)
def test_mnist_rotate_final_layer_invariance(basis_formula, edge_formula, rtol=1e-7, atol=1e-8):
    """Test that the non-final edges are the same for MNIST whether or not we rotate the final
    layer.

    TODO: This doesn't actually enter the block in the test function. Investigate.
    """
    not_rotated_config = get_mnist_config(
        {
            "basis_formula": basis_formula,
            "edge_formula": edge_formula,
            "dtype": "float64",
            "node_layers": ["layers.1", "layers.2"],
        }
    )

    rotate_final_layer_invariance(config_not_rotated=not_rotated_config, rtol=rtol, atol=atol)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.parametrize(
    "basis_formula, edge_formula",
    [
        ("(1-alpha)^2", "functional"),
        ("(1-0)*alpha", "functional"),
        ("(1-alpha)^2", "squared"),
        ("(1-0)*alpha", "squared"),
        ("jacobian", "squared"),
    ],
)
def test_modular_mlp_rotate_final_layer_invariance(
    basis_formula, edge_formula, rtol=1e-7, atol=1e-8
):
    """Test that the non-final edges are the same for ModularMLP whether or not we rotate the final layer."""
    config = get_modular_mlp_config({"basis_formula": basis_formula, "edge_formula": edge_formula})
    rotate_final_layer_invariance(config_not_rotated=config, rtol=rtol, atol=atol)


@pytest.mark.xfail
@pytest.mark.slow
@pytest.mark.parametrize(
    "basis_formula, edge_formula, dtype_str",
    [
        # functional fp32 currently fails with these tolerances
        # ("(1-alpha)^2", "functional", "float32"),
        # ("(1-0)*alpha", "functional", "float32"),
        ("(1-alpha)^2", "functional", "float64"),
        ("(1-0)*alpha", "functional", "float64"),
        ("(1-alpha)^2", "squared", "float32"),
        ("(1-0)*alpha", "squared", "float32"),
        ("(1-alpha)^2", "squared", "float64"),
        ("(1-0)*alpha", "squared", "float64"),
    ],
)
def test_modular_arithmetic_rotate_final_layer_invariance(
    basis_formula,
    edge_formula,
    dtype_str,
    rtol=1e-3,
    atol=1e-3,
):
    """Test that the non-final edges are independent of final layer rotation for modadd.

    Note that atol is necessary as the less important edges do deviate. The largest edges are
    between 1e3 and 1e5 large.
    """
    rotate_final_layer_invariance(
        config_not_rotated=get_modular_arithmetic_config(
            {"basis_formula": basis_formula, "edge_formula": edge_formula, "dtype": dtype_str}
        ),
        rtol=rtol,
        atol=atol,
    )


def test_mnist_build_graph_invalid_node_layers():
    """Test that non-sequential node_layers raises an error."""
    config = get_mnist_config({"node_layers": ["layers.0", "layers.2"]})
    with pytest.raises(AssertionError, match="is not a subsequence of all_node_layers:"):
        graph_build_test(config=config, atol=0)


def test_modular_arithmetic_build_graph_invalid_node_layers():
    """Test that out of order node_layers raises an error."""
    # ln1 should be before mlp_in
    config = get_modular_arithmetic_config({"node_layers": ["mlp_in.0", "ln1.0", "unembed"]})
    with pytest.raises(AssertionError, match="Node layers must be in order."):
        graph_build_test(config=config, atol=0)


@pytest.mark.slow
def test_svd_basis():
    config = get_pythia_config({"basis_formula": "svd"})
    results = rib_build(config)
    for interaction_rotations in results.interaction_rotations:
        assert (interaction_rotations.C is None) == (interaction_rotations.W is None)
        if interaction_rotations.C is not None:
            assert torch.allclose(interaction_rotations.C, interaction_rotations.W, atol=0)


def centered_rib_test(results: RibBuildResults, atol=1e-6):
    """
    Test that centered rib (and pca, aka centered svd) works as expected.

    In particular:
    - We collect the rib activations
    - We collect the means, checking that the mean == 1 exactly at the bias positions
    - We expect there to be a single rib direction (the 'constant' direction or 'const_dir') in
        which the activations is constant for all inputs. This is always returned as the 0th rib
        dir. When written in the neuron basis this direciton will equal to the mean-activation at
        non bias positions and 1/sqrt(# bias positions) at bias positions. C_inv gives us all
        rib directions in the original coordinate system, so we compare C_inv[0] with this expected
        direction. There's also various scaling factors.

    We then check 4 properties:
    1. We assert the `const_dir` written in the original coords matches what we expect
    2. We assert that all other rib directions have no bias component (in the original coordinates)
    3. We assert the rib activation of `const_dir` is always 1
    4. We assert the mean rib activation of all other directions is 0 (they are centered)
    """
    # collect C_inv, rib acts, means, bias positions
    all_rib_acts = get_rib_acts_test(results, atol=1e-6)
    all_mean_acts = get_means(results, atol=atol)
    interaction_rotations = rotation_list_to_dict(results.interaction_rotations)
    # output and pre-unembed have no bias
    m_names = [
        m_name for m_name in results.config.node_layers if m_name not in ["output", "unembed"]
    ]
    for m_name in m_names:
        C_inv = interaction_rotations[m_name].C_pinv  # [rib_dir, emb_pos]
        mean_acts = all_mean_acts[m_name].cpu()  # [emb_pos]
        rib_acts = all_rib_acts[m_name]  # [batch, (seqpos?), rib_dir]

        # compute the expected bias direction
        # we can't directly compare, as there's some scale factor. This involves:
        # - ±1, as rib is allowed to find the opposite direction
        # - a factor 1/sqrt(# bias positions) from compressing the bias directions
        # - scale factors from D and lambda when basis != svd
        if results.config.basis_formula == "svd":
            scale_factor = torch.sign(C_inv[0, -1])
        else:
            # in the non-svd case we just get the scale factor from comparing one component
            # with what we expect.
            # this is less strong as a check, but it's hard to compute D and lambda
            scale_factor = C_inv[0, -1]

        expected_const_dir = mean_acts * scale_factor

        # Check 1: Assert this is close to the actual const direction
        assert_is_close(C_inv[0, :], expected_const_dir, atol=atol, rtol=0, m_name=m_name)

        # Check 2: no other rib dir has non-zero component in the bias position
        assert_is_zeros(C_inv[1:][:, -1], atol=atol, m_name=m_name)

        # Check 3: rib act in the constant direction is actually constant.
        # in particualar it's the inverse of the scaling factor above
        assert_is_close(rib_acts[..., 0], 1 / scale_factor, atol=atol, rtol=0, m_name=m_name)

        # Check 4: all other rib acts are have mean zero across the dataset
        mean_rib_acts = einops.reduce(rib_acts, "... ribdir -> ribdir", "mean")
        assert_is_zeros(mean_rib_acts[1:], atol=atol, m_name=m_name)


@pytest.mark.slow
@pytest.mark.parametrize("basis_formula", ["(1-alpha)^2", "(1-0)*alpha", "svd", "jacobian"])
def test_centered_rib_mnist(basis_formula):
    """Test that centered rib works for MNIST."""
    config = get_mnist_config(
        {"basis_formula": basis_formula, "edge_formula": "functional", "center": True}
    )
    results = rib_build(config)
    centered_rib_test(results, atol=1e-6)


@pytest.mark.slow
def test_centered_rib_pythia():
    """Test that the centered rib works for pythia."""
    config = get_pythia_config({"basis_formula": "(1-0)*alpha", "center": True})
    results = rib_build(config)
    centered_rib_test(results, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("basis_formula", ["(1-alpha)^2", "(1-0)*alpha", "svd", "jacobian"])
def test_centered_rib_modadd(basis_formula):
    """Test that centered rib & pca works for modadd."""
    # we set a lower truncation threshold as there are some directions w/ small eigenvals that
    # violate our assumption. I think this is a precision error that shouldn't be a problem
    # elsewhere. See https://apolloresearchhq.slack.com/archives/C06484S5UF9/p1704966880983049
    config = get_modular_arithmetic_config(
        {
            "basis_formula": basis_formula,
            "edge_formula": "squared",
            "center": True,
            "truncation_threshold": 1e-10,
        }
    )
    results = rib_build(config)
    centered_rib_test(results, atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("basis_formula", ["(1-alpha)^2", "(1-0)*alpha"])
@pytest.mark.parametrize("edge_formula", ["functional", "squared"])
@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
@pytest.mark.parametrize("rotate_final", [True, False])
def test_modular_mlp_diagonal_edges_when_linear(
    basis_formula, edge_formula, dtype_str, rotate_final, rtol=1e-7, atol=1e-5, gtol=1e-4
):
    """Test that RIB rotates to a diagonal basis when the ModularMLP is linear.

    Args:
        basis_formula: The basis formula to use.
        edge_formula: The edge formula to use.
        dtype_str: The dtype to use.
        rotate_final: Whether to rotate the final node layer.
        rtol: The relative tolerance to use.
        atol: The absolute tolerance to use.
        gtol: The geometric mean and max column/row value scaling tolerance to use.
    """
    config = get_modular_mlp_config(
        {
            "basis_formula": basis_formula,
            "edge_formula": edge_formula,
            "dtype": dtype_str,
            "rotate_final_node_layer": rotate_final,
            "modular_mlp_config": {"activation_fn": "identity"},
        }
    )
    edges = rib_build(config).edges

    rotated_node_layers = config.node_layers[:-1]
    if (not config.rotate_final_node_layer) and config.node_layers[-1] == "output":
        rotated_node_layers = rotated_node_layers[:-1]

    for node_layer_idx in range(len(rotated_node_layers)):
        # assert that all off diagonal entries agree within rtol of 0. Deal appropriately with the
        # case that matrices are not square
        edge_val = edges[node_layer_idx].E_hat
        diag_target = torch.zeros_like(edge_val)
        min_dim = min(edge_val.shape)
        diag_target[:min_dim, :min_dim] = torch.diag(torch.diag(edge_val))
        difference = edge_val - diag_target

        # Check that off-diagonal entries are small relative to the maximum of 1) the largest entry
        # in that entry's row or column (which should be on the diagonal) and 2) the geometric mean
        # of diagonal entries. The geometric mean is used because some of the diagonal entries may
        # be zero (or very close to it) and it is not neccessary that the off diagonal entries are
        # much smaller than even these small diagonal entries.
        max_entries_in_row = edge_val.abs().amax(dim=1, keepdim=True)
        max_entries_in_column = edge_val.abs().amax(dim=0, keepdim=True)
        max_entries = torch.max(max_entries_in_row, max_entries_in_column)
        max_entries = torch.max(max_entries, torch.diag(edge_val).abs().log().mean().exp())
        assert_close(difference / max_entries, torch.zeros_like(difference), rtol=0, atol=gtol)
        # Check off-diagonal edges are much smaller than the largest edge in that layer
        # atol=rtol is correct here, we are measuring a relative value against 0.
        assert_close(
            difference / edge_val.abs().max(), torch.zeros_like(difference), rtol=0, atol=rtol
        )
        # Check off-diagonal edges are somewhat close to zero (absolute)
        assert_close(difference, torch.zeros_like(difference), rtol=0, atol=atol)


@pytest.mark.slow
def test_stochastic_source_single_pos_modadd():
    """Show that modadd after only needs a single stochastic source when using a single position.

    Since there is only a single position dimension after add_resid1 in modadd when setting
    last_pos_module_type='add_resid1', and since our stochastic sources are either -1 or 1 and are
    then squared in the edge formula, we should only need a single source to get exactly the same
    edges as edge_formula=squared.
    """
    node_layers = ["mlp_in.0", "mlp_out.0"]
    # Calc squared edges
    config_squared = get_modular_arithmetic_config(
        {
            "edge_formula": "squared",
            "node_layers": node_layers,
            "last_pos_module_type": "add_resid1",
            "n_stochastic_sources_edges": None,
        }
    )
    squared_edges = rib_build(config_squared).edges[0].E_hat

    # Calc stochastic edges
    config_stochastic = get_modular_arithmetic_config(
        {
            "edge_formula": "stochastic",
            "node_layers": node_layers,
            "last_pos_module_type": "add_resid1",
            "n_stochastic_sources_edges": 1,
        }
    )
    stochastic_edges = rib_build(config_stochastic).edges[0].E_hat

    assert_is_close(squared_edges, stochastic_edges, atol=0, rtol=0)


@pytest.mark.slow
def test_stochastic_source_modadd_convergence():
    """Show that modadd after converges to squared edges as n_stochastic_sources_edges increases.

    We measure the mean absolute difference between the squared edges and the stochastic edges as
    n_stochastic_sources_edges increases. We expect this to decrease monotonically.

    Here, we set last_pos_module_type='unembed' so that we have multiple position dimensions in the
    mlp layer.

    NOTE: This is quite a weak test, but the runs a slow so we're taking a hit on the test quality.
    """
    node_layers = ["mlp_in.0", "mlp_out.0"]
    return_set_n_samples = 3
    batch_size = 3

    # Calc squared edges
    config_squared = get_modular_arithmetic_config(
        {
            "dataset": {"return_set_n_samples": return_set_n_samples},
            "batch_size": batch_size,
            "edge_formula": "squared",
            "node_layers": node_layers,
            "last_pos_module_type": "unembed",
            "n_stochastic_sources_edges": None,
        }
    )
    squared_edges = rib_build(config_squared).edges[0].E_hat

    # Calc stochastic edges
    all_stochastic_edges = []
    abs_diffs = []
    # Ideally we'd use more sources, but that is very slow
    for n_stochastic_sources_edges in [1, 3, 7]:
        config_stochastic = get_modular_arithmetic_config(
            {
                "dataset": {"return_set_n_samples": return_set_n_samples},
                "batch_size": batch_size,
                "edge_formula": "stochastic",
                "node_layers": node_layers,
                "last_pos_module_type": "unembed",
                "n_stochastic_sources_edges": n_stochastic_sources_edges,
            }
        )
        stochastic_edges = rib_build(config_stochastic).edges[0].E_hat
        all_stochastic_edges.append(stochastic_edges)
        abs_diffs.append(torch.abs(stochastic_edges - squared_edges).mean())

    # Check that the mean absolute differences decrease as n_stochastic_sources_edges increases.
    assert (
        torch.diff(torch.tensor(abs_diffs)) <= 0
    ).all(), "abs_diffs is not monotonically decreasing"
    # Check that the final relative and absolute differences are small.
    assert_is_close(all_stochastic_edges[-1], squared_edges, atol=1e1, rtol=1e-5)
