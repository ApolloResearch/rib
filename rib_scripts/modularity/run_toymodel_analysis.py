"""
A script to run a first pass of modularity analysis on a RIB build and save various helpful related
plots. Will likely require some customization for particular use cases -- you are encouraged to
copy the code and modify it to your needs.

Example usage:

```bash
python run_modularity.py /path/to/rib_build.pt [--threshold 0.2] [--gamma 1.0] [--plot_edge_dist]
   [--plot_piano] [--plot_graph] [--log_norm] [--ablation_path /path/to/ablation_results.json]
"""

from pathlib import Path
from typing import Optional, Union

import fire
import matplotlib.pyplot as plt
import torch

from rib.ablations import AblationConfig, BisectScheduleConfig, load_bases_and_ablate
from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.log import logger
from rib.modularity import ByLayerLogEdgeNorm, EdgeNorm, GraphClustering, SqrtNorm
from rib.rib_builder import RibBuildResults


def run_bisect_ablation(results_path: Union[str, Path], threshold=0.2):
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    assert results.config.dataset is not None, "No dataset found in results."
    assert isinstance(
        results.config.dataset,
        ModularArithmeticDatasetConfig | HFDatasetConfig | VisionDatasetConfig,
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
        dataset=results.config.dataset,
        ablation_node_layers=results.config.node_layers,
        batch_size=results.config.gram_batch_size or results.config.batch_size,
        dtype="float64",
        seed=0,
        eval_type="accuracy",
    )
    load_bases_and_ablate(config)


def run_modularity(
    results_path: Union[str, Path],
    threshold: float = 0.1,
    gamma: float = 1.0,
    plot_piano: bool = True,
    plot_graph: bool = True,
    log_norm: bool = True,
    ablation_path: Optional[Union[str, Path]] = None,
):
    """
    This function runs modularity analysis on a RIB build and saves various helpful related plots.


    Args:
        results_path: The path to the RIB build results file.
        threshold: The loss increase threshold for the bisect ablation.
        gamma: The resolution parameter for the modularity analysis.
        plot_edge_dist: Whether to plot the edge distribution ecdfs. Supported only for multi-layer
            models with node_layers in ["ln1.0", "ln1_out.0", "attn_in.0", "ln2.0", "ln2_out.0",
            "mlp_in.0"].
        plot_piano: Whether to plot a piano plot, giving an overview of modular structure found.
        plot_graph: Whether to plot the full RIB graph with nodes colored and sorted by module.
        log_norm: Whether to use a log edge norm. If True, will need to use the output of an edge
            ablation experiment to determine the epsilon cutoffs. If no ablation file is found,
            will first run the ablation experiment then run the modularity analysis. If False will
            use SqrtNorm.
        ablation_path: The path to some bised edge ablation results file. If not provided, will
            pick a default path in the same directory as results_path. Ignored if log_norm is False.
            If not found, will run the ablation experiment.
    """
    results_path = Path(results_path)
    results = RibBuildResults(**torch.load(results_path))
    name_prefix = f"{results.exp_name}-delta{threshold}"

    # Get correct norm
    if log_norm:
        ablation_path = (
            Path(ablation_path)
            if ablation_path is not None
            else results_path.parent / f"{name_prefix}-bisect_edge_ablation_results.json"
        )
        if not ablation_path.exists():
            logger.info("Ablation file not found, running ablation...")
            run_bisect_ablation(results_path, threshold)

        edge_norm: EdgeNorm = ByLayerLogEdgeNorm.from_bisect_results(ablation_path, results)
    else:
        edge_norm = SqrtNorm()

    logger.info(f"Making RIB graph in networkit & running clustering...")
    graph = GraphClustering(
        results, edge_norm, gamma=gamma, node_layers=["ln1.0", "ln2.0", "unembed"]
    )
    logger.info(f"Finished clustering.")

    # Piano plot
    if plot_piano:
        graph.paino_plot()
        paino_path = results_path.parent / f"{name_prefix}-gamma{gamma}-paino.png"
        plt.suptitle(
            f"{results.exp_name}\nNonsingleton clusters for threshold={threshold}, gamma={gamma}"
        )
        plt.savefig(paino_path)
        logger.info(f"Saved paino plot to {paino_path.absolute()}")
        plt.clf()

    # TODO Graph plot
    # if plot_graph:
    #     rib_graph_path = results_path.parent / f"{name_prefix}-gamma{gamma}-graph.png"
    #     plot_modular_graph(graph=graph, out_file=rib_graph_path)


if __name__ == "__main__":
    fire.Fire(run_modularity)
