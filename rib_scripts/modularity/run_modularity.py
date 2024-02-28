import warnings
from pathlib import Path
from typing import Optional, Union

import fire
import matplotlib.pyplot as plt
import torch

from rib.ablations import AblationConfig, BisectScheduleConfig, load_bases_and_ablate
from rib.data import HFDatasetConfig
from rib.log import logger
from rib.modularity import AdaptiveEdgeNorm, RIBGraph
from rib.rib_builder import RibBuildResults
from rib.utils import replace_pydantic_model
from rib_scripts.rib_build.plot_graph import plot_by_layer


def _num_layers(results: RibBuildResults) -> int:
    return max(int(nl.split(".")[1]) for nl in results.config.node_layers if "." in nl)


def edge_distribution(
    results,
    layout: Optional[tuple[int, int]] = None,
    xlim=(None, None),
    ylim=(0, 1),
    vlines: Optional[dict[str, float]] = None,
):
    """Plots culmulative distribution functions of the edge values.

    Helpful for undetstanding the epsilon cutoffs used for edge normalization, especially with
    `edge_distribution(..., vlines=AdaptiveEdgeNorm.eps_by_layer)`.

    Args:
        results: The results from a RIB build.
        layout: The number of rows and columns of the plot.
        xlim: The x-axis limits.
        ylim: The y-axis limits.
        vlines: A dictionary of node_layer -> float, drawn as vertical lines on the subplots.
    """
    layout = layout or (_num_layers(results) // 4 + 1, 4)
    ps = torch.cat([torch.linspace(0.02, 0.9, 70), torch.linspace(0.9, 1, 150)])
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
        }

    for edge in results.edges:
        in_nl = edge.in_node_layer
        if in_nl in ["ln_final", "ln_final_out"]:
            continue
        prefix, layer = in_nl.split(".")
        E = edge.E_hat.to(torch.float32)
        xs = torch.quantile(E, ps).cpu().numpy()
        ax = axs.flat[int(layer)]
        color = get_prefix_colors()[prefix]
        ax.plot(xs, ps, label=in_nl, color=color)
        ax.set_title(f"layer {layer}")

        if vlines is not None:
            ax.axvline(vlines[in_nl], color=color, linestyle="--")

    for ax in axs.flat[len(results.edges) :]:
        ax.axis("off")

    plt.xscale("log")
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    axs.flat[-1].legend()
    plt.tight_layout()


def run_bisect_ablation(results_path: Union[str, Path], threshold=0.2):
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    assert isinstance(results.config.dataset, HFDatasetConfig)  # for mypy
    dataset_return_sets = {
        "roneneldan/TinyStories": "validation",
        "NeelNanda/pile-10k": "train",  # no test
    }
    dataset = replace_pydantic_model(
        results.config.dataset,
        {
            "return_set": dataset_return_sets[results.config.dataset.name],
            "n_samples": 10,
        },
    )
    config = AblationConfig(
        exp_name=f"{results.exp_name}-bisect-{threshold}",
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
    threshold: float = 0.2,
    gamma: float = 1.0,
    plot_edge_dist: bool = True,
    plot_normalized_rib_graph: bool = True,
    plot_piano: bool = True,
):
    """
    This function runs modularity analysis on a RIB build and saves various helpful related plots.
    """
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    ablation_path = (
        results_path.parent / f"{results.exp_name}-bisect-{threshold}_edge_ablation_results.json"
    )
    if not ablation_path.exists():
        logger.info("Ablation file not found, running ablation...")
        run_bisect_ablation(results_path, threshold)

    edge_norm = AdaptiveEdgeNorm.from_bisect_results(ablation_path, results)

    if plot_edge_dist:
        edge_distribution(results, vlines=edge_norm.eps_by_layer)
        edge_dist_path = results_path.parent / f"edge_distribution_{threshold}.png"
        plt.savefig(edge_dist_path)
        logger.info(f"Saved edge distribution plot to {edge_dist_path.absolute()}")

        # erronius warning, see https://github.com/matplotlib/matplotlib/issues/9970
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempt to set non-positive")
            plt.clf()

    if plot_normalized_rib_graph:
        rib_graph_path = results_path.parent / f"rib_graph_{threshold}.png"
        plot_by_layer(results, edge_norm=edge_norm, out_file=rib_graph_path)
        plt.clf()

    logger.info(f"Making RIB graph in networkit")
    graph = RIBGraph(results, edge_norm)
    logger.info(f"Running clustering")
    graph.run_leiden(gamma=gamma)

    if plot_piano:
        graph.paino_plot()
        paino_path = results_path.parent / f"paino-lossdiff_{threshold}-gamma_{gamma}.png"
        plt.suptitle(f"Nonsingleton clusters for threshold={threshold}, gamma={gamma}")
        plt.savefig(paino_path)
        logger.info(f"Saved paino plot to {paino_path.absolute()}")

    # logger.info(
    #     f"Plots saved:\nEdge Dist: {edge_dist_path.absolute()}\nRIB Graph: {rib_graph_path.absolute()}\nPaino: {paino_path.absolute()}"
    # )


if __name__ == "__main__":
    fire.Fire(run_modularity)
