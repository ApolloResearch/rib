"""Script for comparing edges from graph files
"""
import fire
import matplotlib.pyplot as plt
import torch


def compare_edges(*result_files):
    """Compare edges from graph files

    Args:
        result_files: Paths to the graph files to compare.
    """
    n_cols = len(result_files)
    # Open first file
    test_file = torch.load(result_files[0])
    # Get edges
    test_edges = test_file["edges"]
    n_rows = len(test_edges)
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    # Loop over rows
    for row in range(n_rows):
        test_name_in = test_edges[row]["in_node_layer"]
        test_name_out = test_edges[row]["out_node_layer"]
        test_vals = test_edges[row]["E_hat"]
        for column in range(n_cols):
            result = torch.load(result_files[column])
            assert test_name_in == result["edges"][row]["in_node_layer"]
            assert test_name_out == result["edges"][row]["out_node_layer"]
            # edges = result["edges"][row]["E_hat"] - test_vals
            edges = torch.log10(result["edges"][row]["E_hat"])
            vmax = torch.log10(edges[0, 0] * 10)
            vmin = vmax - 5
            im = axes[row, column].imshow(edges[:50, :50], vmin=vmin, vmax=vmax)
            axes[row, column].set_title(result_files[column].split("/")[-1][-13:], fontsize=8)
            fig.colorbar(im, ax=axes[row, column])


compare_edges(
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic_rib_graph.pt",
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic-stochastic_0_rib_graph.pt",
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic-stochastic_1_rib_graph.pt",
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic-stochastic_5_rib_graph.pt",
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic-stochastic_100_rib_graph.pt",
    "/mnt/ssd-interp/stefan/rib/rib_scripts/rib_build/out/modular_arithmetic-stochastic_500_rib_graph.pt",
)

# if __name__ == "__main__":
#     fire.Fire(compare_edges, serialize=lambda _: "")
