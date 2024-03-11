"""
TODO
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch

from rib.ablations import AblationConfig, BisectScheduleConfig, load_bases_and_ablate
from rib.data import HFDatasetConfig
from rib.log import logger
from rib.modularity import ByLayerLogEdgeNorm, EdgeNorm, GraphClustering, SqrtNorm
from rib.rib_builder import RibBuildResults
from rib.utils import replace_pydantic_model
from rib_scripts.rib_build.plot_graph import plot_modular_graph


def _num_layers(results: RibBuildResults) -> int:
    return max(int(nl.split(".")[1]) for nl in results.config.node_layers if "." in nl) + 1


def edge_distribution(
    results: RibBuildResults,
    layout: Optional[tuple[int, int]] = None,
    xlim=(None, None),
    ylim=(None, 1),
    vlines: Optional[dict[str, float]] = None,
):
    """Plots culmulative distribution functions of the edge values.

    Helpful for undetstanding the epsilon cutoffs used for edge normalization, especially with
    `edge_distribution(..., vlines=AdaptiveEdgeNorm.eps_by_layer)`.

    Skips ln_final and ln_final_out layers.

    Args:
        results: The results from a RIB build.
        layout: The number of rows and columns of the plot.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
        vlines: A dictionary of node_layer -> float, drawn as vertical lines on the subplots.
    """
    n_layers = _num_layers(results)
    layout = layout or (n_layers // 4, 4)
    qs = torch.cat([torch.linspace(0.01, 0.9, 100), torch.linspace(0.9, 1, 100)])
    figsize = (layout[1] * 3, layout[0] * 3)
    fig, axs = plt.subplots(*layout, figsize=figsize, sharex=True, sharey=True)

    def get_prefix_colors():
        cmap = plt.get_cmap("Paired")
        return {
            "ln1": cmap.colors[1],
            "ln1_out": cmap.colors[0],
            "attn_in": cmap.colors[3],
            "ln2": cmap.colors[9],
            "ln2_out": cmap.colors[8],
            "mlp_in": cmap.colors[5],
            "mlp_out": cmap.colors[6],
            "unembed": cmap.colors[7],
            "output": cmap.colors[2],
        }

    plt.xscale("log")
    for edge in results.edges:
        in_nl = edge.in_node_layer
        if in_nl in ["ln_final", "ln_final_out"]:
            continue
        prefix, layer = in_nl.split(".") if "." in in_nl else (in_nl, 0)
        E = edge.E_hat.to(torch.float32).cpu().numpy()
        xs = np.quantile(E, qs)  # using np instead of torch as it's much much faster
        ax: plt.Axes = axs.flat[int(layer)]
        color = get_prefix_colors()[prefix]
        ax.plot(xs, qs, label=in_nl, color=color)
        ax.set_title(f"layer {layer}")

        if vlines is not None:
            ax.axvline(vlines[in_nl], color=color, linestyle="--")

    for ax in axs.flat[n_layers + 1 :]:
        ax.remove()

    plt.ylim(*ylim)
    plt.xlim(*xlim)
    axs.flat[n_layers - 1].legend()
    plt.suptitle(f"{results.exp_name} edge ecdf")
    plt.tight_layout()


def run_bisect_ablation(results_path: Union[str, Path], threshold=0.2):
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    assert isinstance(results.config.dataset, HFDatasetConfig)  # for mypy
    dataset = replace_pydantic_model(
        results.config.dataset,
        {
            "return_set_portion": "last",
            "n_samples": 100,
            "n_documents": 100,
        },
    )
    config = AblationConfig(
        exp_name=f"{results.exp_name}-delta{threshold}-bisect",
        out_dir=results_path.parent.absolute(),
        ablation_type="edge",
        rib_results_path=results_path.absolute(),
        schedule=BisectScheduleConfig(
            schedule_type="bisect",
            score_target_difference=threshold,
            scaling="logarithmic",
        ),
        dataset=dataset,
        ablation_node_layers=results.config.node_layers,
        batch_size=results.config.gram_batch_size or results.config.batch_size,
        dtype="float32",
        seed=0,
        eval_type="ce_loss",
    )
    load_bases_and_ablate(config)


def run_modularity(
    results_path: Union[str, Path],
    threshold: float = 0.1,
    gamma: float = 1.0,
    plot_edge_dist: bool = True,
    plot_piano: bool = True,
    plot_graph: bool = True,
    log_norm=True,
):
    """
    This function runs modularity analysis on a RIB build and saves various helpful related plots.
    """
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    name_prefix = f"{results.exp_name}-delta{threshold}"
    if log_norm:
        ablation_path = results_path.parent / f"{name_prefix}-bisect_edge_ablation_results.json"
        if not ablation_path.exists():
            logger.info("Ablation file not found, running ablation...")
            run_bisect_ablation(results_path, threshold)

        edge_norm: EdgeNorm = ByLayerLogEdgeNorm.from_bisect_results(ablation_path, results)
    else:
        edge_norm = SqrtNorm()

    if plot_edge_dist:
        if hasattr(edge_norm, "eps_by_layer"):
            xlim_buffer = 1.5
            xlim = (
                min(edge_norm.eps_by_layer.values()) / xlim_buffer,
                max(edge_norm.eps_by_layer.values()) * xlim_buffer,
            )
            vlines = edge_norm.eps_by_layer
        else:
            xlim = (None, None)
            vlines = None
        edge_distribution(results, vlines=vlines, xlim=xlim)
        edge_dist_path = results_path.parent / f"{name_prefix}-edge_ecdf.png"
        plt.savefig(edge_dist_path)
        logger.info(f"Saved edge distribution plot to {edge_dist_path.absolute()}")

        # erronius warning, see https://github.com/matplotlib/matplotlib/issues/9970
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempt to set non-positive")
            plt.clf()

    logger.info(f"Making RIB graph in networkit & running clustering...")
    graph = GraphClustering(results, edge_norm, gamma=gamma)
    logger.info(f"Finished clustering.")

    if plot_piano:
        graph.paino_plot()
        paino_path = results_path.parent / f"{name_prefix}-gamma{gamma}-paino.png"
        plt.suptitle(
            f"{results.exp_name}\nNonsingleton clusters for threshold={threshold}, gamma={gamma}"
        )
        plt.savefig(paino_path)
        logger.info(f"Saved paino plot to {paino_path.absolute()}")
        plt.clf()

    if plot_graph:
        rib_graph_path = results_path.parent / f"{name_prefix}-gamma{gamma}-graph.png"
        plot_modular_graph(graph=graph, out_file=rib_graph_path)


if __name__ == "__main__":
    fire.Fire(run_modularity)
