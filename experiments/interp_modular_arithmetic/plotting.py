from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


def fft_plot_cos_phase_1d(acts, rtol=1e-4):
    r"""
    Plot the cos phase of a 1D FFT data. The input to the FFT is assumed to be real-valued.

    Derivation for why we can express the FFT terms \tilde A_f e^(i2π/113 f x) as
    A_f cos(2π/113 f x + φ), for real-valued input:

        1. Convert from f=0...112 to f=-56...56 since f=57...112 correspond to f=-56...-1),
           the difference is a expoent shift by 2π.

        2. Convert to cos and sin terms, e^(i2π/113 f x) = cos(2π/113 f x) + i sin(2π/113 f x)

        3. Combine pairs of negative and positive frequencies:
           (A_f + A_-f) cos(2π/113 f x) + (A_f - A_-f) i sin(2π/113 f x)

        4. Use that the ampltiudes of a real-valued FFT are complex conjugates A_f = conj(A_-f).
           (A_f + A_-f) = 2 Re(A_f) and (A_f - A_-f) i = - 2 Im(A_f)

        5. Finally combine cos and sin terms using "harmonic addition"
           (https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Sine_and_cosine):
           a*cos(x) + b*sin(x) = c*cos(x+phi) with c=sign(a)*sqrt(a^2+b^2) and phi=arctan(-b/a)
           or a*cos(x) + b*sin(x) = c*cos(x+phi) with c=sqrt(a^2+b^2) and phi=-atan2(b, a).

           Each f term is sqrt(Re(2A_f)^2 + Im(2A_f)^2) cos(2π/113 f x -atan2(Im(-A_f), Re(A_f)))
           or simply 2*|A_f| cos(2π/113 f x + atan2(Im(A_f), Re(A_f))).

    Args:
        acts: nD tensor of activations, complex. 0th dimension assumed to be frequency.

    Returns:
        fig, ax: matplotlib figure and axis objects.
    """
    # Calculate the FFT
    n_freqs = acts.shape[0]
    assert n_freqs % 2 == 1, f"n_freqs={n_freqs} must be odd"

    # Apply fftshift, makes thinking about the frequencies easier
    fftshift_freqs = torch.fft.fftshift(torch.fft.fftfreq(n_freqs))
    fftshift_amplitudes = torch.fft.fftshift(acts)

    # Central frequency (constant)
    i_center = n_freqs // 2
    assert fftshift_freqs[i_center] == 0, f"fftshift_freqs[center] = {fftshift_freqs[i_center]}"

    # Collect the frequencies and compute the amplitudes and phases
    freq_labels = []
    freq_amplitudes = []
    freq_phases = []
    for i in np.arange(n_freqs // 2, n_freqs):
        f = fftshift_freqs[i]
        if i == i_center:
            a = fftshift_amplitudes[i]
            assert f == 0, "central frequency must be 0"
            assert torch.all(torch.arctan(a.imag / a.real) < rtol), "central frequency must be real"
            freq_labels.append("const")
            freq_amplitudes.append(a.real)
            freq_phases.append(torch.zeros_like(a.real))
        else:
            assert f > 0, f"should be iterating over positive frequencies but f={f}"
            i_pos = i
            i_neg = i_center - (i - i_center)
            a_pos = fftshift_amplitudes[i_pos]
            a_neg = fftshift_amplitudes[i_neg]
            # Assert complex conjugate of amplitudes (due to real input)
            assert torch.allclose(
                a_pos, torch.conj(a_neg), rtol=rtol
            ), "amplitude pairs must be complex conjugates (real input?)"
            assert (n_freqs * f + 0.5) % 1 - 0.5 < rtol, f"{n_freqs}*f={n_freqs*f} must be integer"
            freq_labels.append(f"cos{n_freqs*f:.0f}")
            freq_amplitudes.append(2 * torch.sqrt(a_pos.real**2 + a_pos.imag**2))
            freq_phases.append(-torch.atan2(a_pos.imag, a_pos.real))

    # Plot the amplitudes and phases
    freq_labels = np.array(freq_labels)
    freq_labels_ticks = torch.arange(len(freq_labels))
    freq_amplitudes = np.array(freq_amplitudes)
    freq_phases = np.array(freq_phases)
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    freq_labels_ticks_nD = (
        freq_labels_ticks
        if acts.ndim == 1
        else torch.repeat_interleave(freq_labels_ticks, acts.shape[1])
    )
    im = ax.scatter(
        freq_labels_ticks_nD,
        freq_amplitudes,
        c=freq_phases,
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        s=10,
    )
    ax.plot(freq_labels_ticks, freq_amplitudes, lw=0.5)
    ax.set_xticks(np.arange(len(freq_labels)))
    ax.set_xticklabels(freq_labels, rotation=-90)
    ax.set_ylabel("Amplitude A")
    ax.set_xlabel("Each term A cos f = A*cos(2π/113 f + φ)")
    cbar = fig.colorbar(im, ax=ax, label="Phase φ")
    ax.grid(color="grey", linestyle="--", alpha=0.3)
    return fig, ax


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
