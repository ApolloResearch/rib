from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


def annotated_fft_line_plot(
    acts,
    ax=None,
    fftshift=True,
    title="Default title",
    figsize=(8, 8),
    xlabel="Fourier frequency",
    ylabel="Magnitude",
    label=None,
    annotation_magnitude_threshold=0.05,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    freqs = torch.fft.fftfreq(acts.shape[0])
    if fftshift:
        freqs = torch.fft.fftshift(freqs)
        acts = torch.fft.fftshift(acts)
    phase = acts.angle()
    acts = acts.abs()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(freqs, acts, label=label)
    for j, v in enumerate(acts):
        if v > annotation_magnitude_threshold * acts.max():
            sign = "+" if phase[j] >= 0 else "-"
            ax.text(
                freqs[j],
                min(v + 0.05 * acts.max(), 0.95 * acts.max()),
                f"@{freqs[j]:.3f}:\nMag {v:.1e}\nPhase {phase[j]/np.pi:.2f}π",
                rotation=0,
                fontsize=10,
            )


def plot_activations(acts, title="Default title", nrows=2, ncols=2, figsize=(8, 8), center=True):
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    """Plot the first ncols*nrows activation dimensions as a function of x and y, in a grid of subplots

    Args:
        acts (torch.Tensor): The activations to plot
        title (str, optional): The title of the plot.
        nrows (int, optional): The number of rows to plot. Defaults to 2.
        ncols (int, optional): The number of columns to plot. Defaults to 2.
        figsize (tuple, optional): The size of the figure. Defaults to (8, 8).
        center (bool, optional): Whether to center the activations around zero. Defaults to True.
    """
    fig.suptitle(title)
    for i, axs in enumerate(axes):
        for j, ax in enumerate(axs):
            ax.set_title(f"({i * ncols + j})")
            if center:
                vminmax = acts[:, :, i * ncols + j].abs().max().item()
                kwargs = dict(cmap="RdBu", vmin=-vminmax, vmax=vminmax)
            else:
                kwargs = dict(cmap="viridis")
            im = ax.imshow(
                acts[:, :, i * ncols + j].numpy(),
                aspect="equal",
                origin="lower",
                **kwargs,
            )
            fig.colorbar(im, ax=ax)

    datetimestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = datetimestr + (
        title.replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("'", "")
    )
    save_path = Path(__file__).parent.joinpath(f"out/{filename}.png")
    plt.savefig(save_path)


def plot_fft_activations(
    acts,
    title="Default title",
    nrows=4,
    figsize=(8, 16),
    annotate=False,
    fftshift=True,
    phaseplot_magnitude_threshold=0.5,
):
    """Plot the first nrows activation dimensions as a function of x and y, in a grid of subplots.

    The first column is the magnitude, the second column is the phase.

    Args:
        acts (torch.Tensor): The activations to plot
        title (str, optional): The title of the plot.
        nrows (int, optional): The number of rows to plot. Defaults to 4.
        figsize (tuple, optional): The size of the figure. Defaults to (8, 16).
        annotate (bool, optional): Whether to annotate the phase plot with the magnitude and phase
        fftshift (bool, optional): Whether to fftshift the activations before plotting. Defaults
            to True.
        phaseplot_magnitude_threshold (float, optional): The magnitude threshold deciding
            whether to plot the phase. Defaults to 0.5.
    """
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    freqs = torch.fft.fftfreq(acts.shape[0])
    if fftshift:
        freqs = torch.fft.fftshift(freqs).numpy()
        acts = torch.fft.fftshift(acts, dim=[0, 1])
        title += "(fftshift'ed)"
        extent = [freqs[0], freqs[-1], freqs[0], freqs[-1]]
    else:
        extent = None

    fig.suptitle(title)
    for col, title_info in enumerate(["magnitude", "phase"]):
        for row, ax in enumerate(axes[:, col]):
            vminmax = acts.abs()[:, :, row].max().item()
            # abs_norm = SymLogNorm(linthresh=vminmax / 10, linscale=1, vmin=-vminmax, vmax=vminmax)
            abs_norm = LogNorm(vmin=vminmax / 1e2, vmax=vminmax, clip=True)
            if col == 0:
                im = ax.imshow(
                    acts.abs()[:, :, row].numpy(),
                    cmap="Greys",
                    norm=abs_norm,
                    aspect="equal",
                    extent=extent,
                    origin="lower",
                )
                fig.colorbar(im, ax=ax)
            else:
                phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
                # Use scatter
                im = ax.scatter(
                    np.repeat(freqs, acts.shape[0]),
                    np.tile(freqs, acts.shape[0]),
                    c=acts.angle()[:, :, row].numpy().flatten(),
                    cmap="hsv",
                    norm=phase_norm,
                    alpha=(
                        abs_norm(acts.abs()[:, :, row].numpy()).flatten()
                        > phaseplot_magnitude_threshold
                    ),
                    edgecolors="black",
                )
                if annotate:
                    # Annotate scatter
                    for i, xfreq in enumerate(freqs):
                        for j, yfreq in enumerate(freqs):
                            val = acts.abs()[i, j, row].numpy()
                            phase = acts.angle()[i, j, row].numpy()
                            sign = "+" if phase >= 0 else "-"
                            if abs_norm(val) > phaseplot_magnitude_threshold:
                                ax.text(
                                    xfreq,
                                    yfreq,
                                    f"@({xfreq:.3f}, {yfreq:.3f})\nMag {val:.1e}\nPhase {phase/np.pi:.2f}π",
                                    fontsize=4,
                                    ha="center",
                                    va="center",
                                    path_effects=[pe.withStroke(linewidth=0.5, foreground="white")],
                                )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
                cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            ax.set_title(f"Dim {row}, {title_info}")

    datetimestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = datetimestr + (
        title.replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("'", "")
    )
    dpi = 600 if annotate else 300
    save_path = Path(__file__).parent.joinpath(f"out/{filename}.png")
    plt.savefig(save_path, dpi=dpi)
