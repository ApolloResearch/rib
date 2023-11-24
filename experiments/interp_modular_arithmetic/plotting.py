from colorsys import hls_to_rgb
from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


# Neel's helper functions
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        tensor.numpy(),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(tensor.numpy(), labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = x.numpy()
    y = y.numpy()
    px.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show(renderer)


# Stefan's helper functions
def _colorize(r, angle, vmin=0, vmax=None):
    """
    Convert magnitude and phase to color  for plotting, brightness (white=0) and hue (real=?).

    Input shapes [x, y]. Used for 2D FFT imshows, transpose included.
    """
    vmax = r.max() if vmax is None else vmax
    pi = np.pi
    h = (angle + pi) / (2 * pi)  # (x, y)
    # 1 = white, minimum to 1-1/1.5 = dark but not black
    l = 1 - (r - vmin) / (vmax - vmin) / 1.5  # (x, y)
    s = 0.8
    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # (3,x,y)
    c = c.swapaxes(0, 2)  # (y,x,3) -- this combines transpose and move color to 3rd dim!
    return c  # Imshow wants [y, x, rgb]


def _fftshift_1d(acts, skip_warning=False):
    n_freqs = acts.shape[0]
    if acts.ndim == 2 and acts.shape[1] == n_freqs and not skip_warning:
        raise Warning("acts is square, sure this is 1D?")
    freqs = torch.fft.fftfreq(n_freqs)
    freqs_shifted = torch.fft.fftshift(freqs)
    acts_shifted = torch.fft.fftshift(acts, dim=0)
    return freqs_shifted, acts_shifted


def _fftshift_2d(acts):
    n_freqs = acts.shape[0]
    assert acts.shape[1] == n_freqs, "acts must be square"
    freqs = torch.fft.fftfreq(n_freqs)
    freqs_shifted = torch.fft.fftshift(freqs)
    acts_shifted = torch.fft.fftshift(acts, dim=[0, 1])
    return freqs_shifted, acts_shifted


def __ftshift_to_cosplusminus(acts_shifted):
    """Convert fftshifted acts to the cos(x+y+φ) and cos(x-y+φ) terms.

    For every combination of non-zero frequencies k_x, k_y, there are four terms:
    = A_++ e^(i2π/113 (k_x x + k_y y)) + A_+- e^(i2π/113 (k_x x - k_y y))
    + A_-+ e^(i2π/113 (-k_x x + k_y y)) + A_-- e^(i2π/113 (-k_x x - k_y y))

    These can be combined into two real terms using A_++ = conj(A_--), A_+- = conj(A_-+):
    = 2*A_++.real cos(2π/113 (k_x x + k_y y)) - 2*A_++.imag sin(2π/113 (k_x x + k_y y))
    + 2*A_+-.real cos(2π/113 (k_x x - k_y y)) - 2*A_+-.imag sin(2π/113 (k_x x - k_y y))

    And these terms can again be combined into two terms using harmonic addition:
    = 2*|A_++| cos(2π/113 (k_x x + k_y y) + atan2(2*A_++.imag, 2*A_++.real))
    + 2*|A_+-| cos(2π/113 (k_x x - k_y y) + atan2(2*A_+-.imag, 2*A_+-.real))

    For the zero-frequency terms, say k_y = 0 WLOG, we have A_+0 = conj(A_-0) and
    = A_+0 e^(i2π/113 (k_x x)) + A_-0 e^(i2π/113 (-k_x x))
    = 2*A_+0.real cos(2π/113 (k_x x)) - 2*A_+0.imag sin(2π/113 (k_x x))
    = 2*|A_+0| cos(2π/113 (k_x x) + atan2(2*A_+0.imag, 2*A_+0.real))
    and for k_x = 0 we have
    = 2*|A_0+| cos(2π/113 (k_y y) + atan2(2*A_0+.imag, 2*A_0+.real))
    and for A_00 we have A_00 = real and
    = A_00 e^0

    So we can plot a 2D FFT as a 2D plot of x+y and x-y waves with magnitude and phase each, either
    as two panels (x+y and x-y separately), or as a single panel with x+y and x-y interleaved.

    Args:
    """
    n_freqs = acts_shifted.shape[0]
    assert acts_shifted.shape[1] == n_freqs, "acts must be square"
    assert n_freqs % 2 == 1, f"n_freqs={n_freqs} must be odd"
    i_center = n_freqs // 2

    A_pp = acts_shifted[i_center + 1 :, i_center + 1 :]
    A_pm = acts_shifted.flip(dims=(1,))[i_center + 1 :, i_center + 1 :]
    A_0p = acts_shifted[i_center, i_center + 1 :]
    A_p0 = acts_shifted[i_center + 1 :, i_center]
    A_00 = acts_shifted[i_center, i_center]
    # Redundant terms
    # A_mm = acts_shifted.flip(dims=(0, 1))[i_center + 1 :, i_center + 1 :]
    # A_mp = acts_shifted.flip(dims=(0,))[i_center + 1 :, i_center + 1 :]
    # A_0m = acts_shifted.flip(dims=(1,))[i_center, i_center + 1 :]
    # A_m0 = acts_shifted.flip(dims=(0,))[i_center + 1 :, i_center]

    cos_plus = torch.zeros_like(acts_shifted[: i_center + 1, : i_center + 1])
    cos_plus[0, 0] = A_00
    cos_plus[0, 1:] = A_0p
    cos_plus[1:, 0] = A_p0
    cos_plus[1:, 1:] = A_pp
    cos_minus = A_pm

    return cos_plus, cos_minus

    # mag_plus = torch.zeros_like(acts_shifted[: i_center + 1, : i_center + 1])
    # mag_plus[0, 0] = A_00.abs()
    # mag_plus[0, 1:] = A_0p.abs()
    # mag_plus[1:, 0] = A_p0.abs()
    # mag_plus[1:, 1:] = A_pp.abs()
    # mag_minus = A_pm.abs()

    # phase_plus = torch.zeros_like(acts_shifted[: i_center + 1, : i_center + 1])
    # phase_plus[0, 0] = 0
    # phase_plus[0, 1:] = A_0p.angle()  # angle == atan2(im, re)
    # phase_plus[1:, 0] = A_p0.angle()
    # phase_plus[1:, 1:] = A_pp.angle()
    # phase_minus = A_pm.angle()

    # return mag_plus, phase_plus, mag_minus, phase_minus


def _extent_from_acts(acts, modes):
    if acts.shape[0] == modes.shape[0]:
        extent = [modes[0], modes[-1], modes[0], modes[-1]]
    elif acts.shape[0] == modes.shape[0] // 2 + 1:
        extent = [0, modes[-1], 0, modes[-1]]
    elif acts.shape[0] == modes.shape[0] // 2:
        n_freqs = modes.shape[0]
        i_center = n_freqs // 2
        extent = [modes[i_center + 1], modes[-1], modes[i_center + 1], modes[-1]]
    else:
        raise ValueError(f"Cannot infer extent from acts.shape={acts.shape}")
    return extent


def _imshow_complex(
    freqs, acts, ax, mag_colorbar=True, phase_colorbar=False, extent=None, phase=None
):
    """Take in default FFT acts and plot the magnitude and phase in a single imshow plot."""
    assert acts.ndim == 2, "acts must be 2D"
    assert acts.shape[0] == acts.shape[1], "acts must be square"
    assert freqs.ndim == 1, "freqs must be 1D"
    modes = 113 * freqs
    extent = _extent_from_acts(acts, modes) if extent is None else extent
    if phase is None:
        mag = acts.abs()
        phase = acts.angle()
    else:
        assert acts.imag.abs().max() == 0
        assert phase.imag.abs().max() == 0
        assert acts.shape == phase.shape
        mag = acts

    vmax = mag.max()
    im = ax.imshow(
        _colorize(mag, phase, vmax=vmax),
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("f_x")
    ax.set_ylabel("f_y")
    if phase_colorbar:
        sm = plt.cm.ScalarMappable(cmap="hsv", norm=Normalize(vmin=-np.pi, vmax=np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
    if mag_colorbar:
        sm2 = plt.cm.ScalarMappable(cmap="Greys", norm=Normalize(vmin=0, vmax=vmax))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax)
        # Rotate labels 45
        cbar2.ax.set_yticklabels(cbar2.ax.get_yticklabels(), rotation=45)


def fft_plot_cos_phase_1d(acts, rtol=1e-4, labels=True, title=""):
    r"""
    Plot the cos phase of a 1D FFT data. The input to the FFT is assumed to be real-valued.

    Derivation for why we can express the FFT terms B_f e^(i2π/113 f x) as
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

        Note now a 1D FFT translates 113 real numbers into 113 complex numbers (with 1
        discrete symmetry) which we turn into 56*2+1 real numbers (ampltidue & phase + const term).

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
            freq_phases.append(torch.atan2(a_pos.imag, a_pos.real))

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
    if labels:
        for i in range(freq_amplitudes.shape[1]):
            ax.plot(freq_labels_ticks, freq_amplitudes[:, i], lw=0.5, label=f"Dim {i}")
        ax.legend(loc="upper right")
    else:
        ax.plot(freq_labels_ticks, freq_amplitudes, lw=0.5, label=f"Dim {i}")
    ax.set_xticks(np.arange(len(freq_labels)))
    ax.set_xticklabels(freq_labels, rotation=-90)
    ax.set_ylabel("Amplitude A")
    ax.set_xlabel("Each term A cos f = A*cos(2π/113 f + φ)")
    fig.colorbar(im, ax=ax, label="Phase φ")
    fig.suptitle(title)
    ax.grid(color="grey", linestyle="--", alpha=0.3)
    return fig, ax


# def annotated_fft_line_plot(
#     acts,
#     ax=None,
#     fftshift=True,
#     title="Default title",
#     figsize=(8, 8),
#     xlabel="Fourier frequency",
#     ylabel="Magnitude",
#     label=None,
#     annotation_magnitude_threshold=0.05,
# ):
#     if ax is None:
#         _, ax = plt.subplots(figsize=figsize)
#     freqs = torch.fft.fftfreq(acts.shape[0])
#     if fftshift:
#         freqs = torch.fft.fftshift(freqs)
#         acts = torch.fft.fftshift(acts)
#     phase = acts.angle()
#     acts = acts.abs()
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.plot(freqs, acts, label=label)
#     for j, v in enumerate(acts):
#         if v > annotation_magnitude_threshold * acts.max():
#             sign = "+" if phase[j] >= 0 else "-"
#             ax.text(
#                 freqs[j],
#                 min(v + 0.05 * acts.max(), 0.95 * acts.max()),
#                 f"@{freqs[j]:.3f}:\nMag {v:.1e}\nPhase {phase[j]/np.pi:.2f}π",
#                 rotation=0,
#                 fontsize=10,
#             )


def fft_plot_eikx_2d(
    acts_orig,
    title="Default title eikx",
    nrows=2,
    ncols=2,
    figsize=None,
    phase_colorbar=True,
    mag_colorbar=True,
):
    figsize = (ncols * 4, nrows * 4) if figsize is None else figsize
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    freqs_shifted, acts_shifted = _fftshift_2d(acts_orig)
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"Dim {i}")
        _imshow_complex(
            freqs_shifted,
            acts_shifted[:, :, i],
            ax,
            phase_colorbar=phase_colorbar,
            mag_colorbar=mag_colorbar,
        )
    fig.suptitle(title)


def fft_plot_cosplusminus(
    acts_orig,
    title="Default title cosplusminus",
    nrows=2,
    figsize=None,
):
    ncols = 2
    figsize = (ncols * 4, nrows * 4) if figsize is None else figsize
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    freqs_shifted, acts_shifted = _fftshift_2d(acts_orig)
    acts_plus, acts_minus = __ftshift_to_cosplusminus(acts_shifted)
    for row, [ax_plus, ax_minus] in enumerate(axes):
        ax_plus.set_title(f"Dim {row}, cos(f_x x + f_y y + φ)")
        _imshow_complex(freqs_shifted, acts_plus[:, :, row], ax_plus)
        ax_minus.set_title(f"Dim {row}, cos(f_x x - f_y y + φ)")
        _imshow_complex(freqs_shifted, acts_minus[:, :, row], ax_minus)
    fig.suptitle(title)


def fft_plot_coscos_sinsin(
    acts_orig,
    title="Default title cosplusminus",
    nrows=2,
    figsize=None,
):
    ncols = 2
    figsize = (ncols * 4, nrows * 4) if figsize is None else figsize
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=True, figsize=figsize)
    freqs_shifted, acts_shifted = _fftshift_2d(acts_orig)
    acts_plus, acts_minus = __ftshift_to_cosplusminus(acts_shifted)
    mag_coscos = acts_plus.abs()
    mag_coscos[1:, 1:] += acts_minus.abs()
    mag_sinsin = acts_plus.abs()[1:, 1:] - acts_minus.abs()
    for row, [ax_plus, ax_minus] in enumerate(axes):
        ax_plus.set_title(f"Dim {row}, A cos(f_x x + φ_x) * cos(f_y y + φ_y)", fontsize=8)
        _imshow_complex(freqs=freqs_shifted, acts=mag_coscos[:, :, row], ax=ax_plus)
        ax_minus.set_title(f"Dim {row}, B sin(f_x x + φ_x) * sin(f_y y + φ_y)", fontsize=8)
        _imshow_complex(freqs=freqs_shifted, acts=mag_sinsin[:, :, row], ax=ax_minus)
    fig.suptitle(title)

    # TODO phases


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
            # abs_norm = LogNorm(vmin=vminmax / 1e2, vmax=vminmax, clip=True)
            abs_norm = Normalize(vmin=0, vmax=vminmax)
            if col == 0:
                im = ax.imshow(
                    acts.abs()[:, :, row].numpy().T,
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


def plot_fft_activations_cosphase(
    acts_in,
    title="Default title",
    nrows=4,
    figsize=(8, 16),
    annotate=False,
    fftshift=True,
    phaseplot_magnitude_threshold=0.5,
):
    """Plot the first nrows activation dimensions as a function of x and y, in a grid of subplots.

    For every combination of non-zero frequencies k_x, k_y, there are four terms:
    = A_++ e^(i2π/113 (k_x x + k_y y)) + A_+- e^(i2π/113 (k_x x - k_y y))
    + A_-+ e^(i2π/113 (-k_x x + k_y y)) + A_-- e^(i2π/113 (-k_x x - k_y y))

    These can be combined into two real terms using A_++ = conj(A_--), A_+- = conj(A_-+):
    = 2*A_++.real cos(2π/113 (k_x x + k_y y)) - 2*A_++.imag sin(2π/113 (k_x x + k_y y))
    + 2*A_+-.real cos(2π/113 (k_x x - k_y y)) - 2*A_+-.imag sin(2π/113 (k_x x - k_y y))

    And these terms can again be combined into two terms using harmonic addition:
    = 2*|A_++| cos(2π/113 (k_x x + k_y y) + atan2(2*A_++.imag, 2*A_++.real))
    + 2*|A_+-| cos(2π/113 (k_x x - k_y y) + atan2(2*A_+-.imag, 2*A_+-.real))

    # Alternatively, we can combine the exp terms the other way, adding up vertically:


    We could also convert this to cos cos, cos sin, sin cos, sin sin terms without phases.

    For the zero-frequency terms, say k_y = 0 WLOG, we have A_+0 = conj(A_-0) and
    = A_+0 e^(i2π/113 (k_x x)) + A_-0 e^(i2π/113 (-k_x x))
    = 2*A_+0.real cos(2π/113 (k_x x)) - 2*A_+0.imag sin(2π/113 (k_x x))
    = 2*|A_+0| cos(2π/113 (k_x x) + atan2(2*A_+0.imag, 2*A_+0.real))
    and for k_x = 0 we have
    = 2*|A_0+| cos(2π/113 (k_y y) + atan2(2*A_0+.imag, 2*A_0+.real))
    and for A_00 we have A_00 = real and
    = A_00 e^0

    So we can plot a 2D FFT as a 2D plot of x+y and x-y waves with magnitude and phase each, either
    as two panels (x+y and x-y separately), or as a single panel with x+y and x-y inbterleaved.

    Args:
    """
    n_freqs = acts_in.shape[0]
    assert acts_in.shape[1] == n_freqs, "acts must be square"
    assert n_freqs % 2 == 1, f"n_freqs={n_freqs} must be odd"
    freqs = torch.fft.fftshift(torch.fft.fftfreq(n_freqs))
    freqs_plus = freqs[n_freqs // 2 + 1 :]
    i_center = n_freqs // 2

    fig, axes = plt.subplots(nrows, ncols=2, constrained_layout=True, figsize=figsize)
    for row in range(nrows):
        acts = torch.fft.fftshift(acts_in[:, :, row], dim=[0, 1])
        A_pp = acts[i_center + 1 :, i_center + 1 :]
        # A_mm = acts.flip(dims=(0, 1))[i_center + 1 :, i_center + 1 :]
        A_pm = acts.flip(dims=(1,))[i_center + 1 :, i_center + 1 :]
        # A_mp = acts.flip(dims=(0,))[i_center + 1 :, i_center + 1 :]
        A_0p = acts[i_center, i_center + 1 :]
        A_p0 = acts[i_center + 1 :, i_center]
        # A_0m = acts.flip(dims=(1,))[i_center, i_center + 1 :]
        # A_m0 = acts.flip(dims=(0,))[i_center + 1 :, i_center]
        A_00 = acts[i_center, i_center]

        # Make an array like this:
        # A00 A0p
        # Ap0 App
        mag_array = torch.zeros((i_center + 1, i_center + 1))
        mag_array[0, 0] = A_00.abs()
        mag_array[0, 1:] = A_0p.abs()
        mag_array[1:, 0] = A_p0.abs()
        mag_array[1:, 1:] = A_pp.abs()
        mag_array2 = A_pm.abs()

        phase_array = torch.zeros((i_center + 1, i_center + 1))
        phase_array[0, 0] = 0
        phase_array[0, 1:] = A_0p.angle()  # angle = atan2(im, re)
        phase_array[1:, 0] = A_p0.angle()
        phase_array[1:, 1:] = A_pp.angle()
        phase_array2 = A_pm.angle()
        vmax = max(mag_array.max(), mag_array2.max())
        axes[row, 0].imshow(
            _colorize(mag_array, phase_array, vmax=vmax),
            extent=[0, i_center, 0, i_center],
            origin="lower",
        )

        axes[row, 1].imshow(
            _colorize(mag_array2, phase_array2, vmax=vmax),
            extent=[1, i_center, 1, i_center],
            origin="lower",
        )
        axes[row, 0].set_ylabel(f"k_y")
        print("mag_array shape", mag_array.shape)
        axes[row, 0].text(0.5, 0.5, f"Max val: {mag_array.max():.2e}")
        axes[row, 0].grid(color="grey", linestyle="--", alpha=0.3)
        axes[row, 0].set_xticks(np.arange(i_center + 2))
        axes[row, 0].set_xticklabels(
            [f"cos{k:.0f}" for k in [0, *113 * freqs_plus]] + [" "], rotation=-90, fontsize=6
        )
        axes[row, 0].set_yticks(np.arange(i_center + 2))
        axes[row, 1].set_xticks(np.arange(1, i_center + 2))
        axes[row, 1].set_yticks(np.arange(1, i_center + 2))

        axes[row, 1].grid(color="grey", linestyle="--", alpha=0.3)
    axes[-1, 0].set_xlabel(f"k_x")
    axes[-1, 1].set_xlabel(f"k_x")
    axes[0, 0].set_title("A+ cos(2π/113 (kx x + ky y) + φ)")
    axes[0, 1].set_title("A- cos(2π/113 (kx x - ky y) + φ)")


def plot_fft_activations_coscos(
    acts_in,
    title="Default title",
    nrows=4,
    figsize=(8, 16),
    annotate=False,
    fftshift=True,
    phaseplot_magnitude_threshold=0.5,
):
    """Plot the first nrows activation dimensions as a function of x and y, in a grid of subplots.

    For every combination of non-zero frequencies k_x, k_y, there are four terms:
    = A_++ e^(i2π/113 (k_x x + k_y y)) + A_+- e^(i2π/113 (k_x x - k_y y))
    + A_-+ e^(i2π/113 (-k_x x + k_y y)) + A_-- e^(i2π/113 (-k_x x - k_y y))

    These can be combined into two real terms using A_++ = conj(A_--), A_+- = conj(A_-+):
    = 2*A_++.real cos(2π/113 (k_x x + k_y y)) - 2*A_++.imag sin(2π/113 (k_x x + k_y y))
    + 2*A_+-.real cos(2π/113 (k_x x - k_y y)) - 2*A_+-.imag sin(2π/113 (k_x x - k_y y))

    And these terms can again be combined into two terms using harmonic addition:
    = 2*|A_++| cos(2π/113 (k_x x + k_y y) + atan2(2*A_++.imag, 2*A_++.real))
    + 2*|A_+-| cos(2π/113 (k_x x - k_y y) + atan2(2*A_+-.imag, 2*A_+-.real))

    # Alternatively, we can combine the exp terms the other way, adding up vertically:


    We could also convert this to cos cos, cos sin, sin cos, sin sin terms without phases.

    For the zero-frequency terms, say k_y = 0 WLOG, we have A_+0 = conj(A_-0) and
    = A_+0 e^(i2π/113 (k_x x)) + A_-0 e^(i2π/113 (-k_x x))
    = 2*A_+0.real cos(2π/113 (k_x x)) - 2*A_+0.imag sin(2π/113 (k_x x))
    = 2*|A_+0| cos(2π/113 (k_x x) + atan2(2*A_+0.imag, 2*A_+0.real))
    and for k_x = 0 we have
    = 2*|A_0+| cos(2π/113 (k_y y) + atan2(2*A_0+.imag, 2*A_0+.real))
    and for A_00 we have A_00 = real and
    = A_00 e^0

    So we can plot a 2D FFT as a 2D plot of x+y and x-y waves with magnitude and phase each, either
    as two panels (x+y and x-y separately), or as a single panel with x+y and x-y inbterleaved.

    Args:
    """
    n_freqs = acts_in.shape[0]
    assert acts_in.shape[1] == n_freqs, "acts must be square"
    assert n_freqs % 2 == 1, f"n_freqs={n_freqs} must be odd"
    freqs = torch.fft.fftshift(torch.fft.fftfreq(n_freqs))
    freqs_plus = freqs[n_freqs // 2 + 1 :]
    i_center = n_freqs // 2

    def colorize(r, angle, vmin=0, vmax=None):
        vmax = r.max() if vmax is None else vmax
        pi = np.pi
        h = (angle + pi) / (2 * pi)
        # l=1 for r=vmin, l=0 for r=vmax
        l = 1 - (r - vmin) / (vmax - vmin)
        s = 0.8
        c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
        c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
        c = c.swapaxes(0, 2)
        return c

    fig, axes = plt.subplots(nrows, ncols=2, constrained_layout=True, figsize=figsize)
    for row in range(nrows):
        acts = torch.fft.fftshift(acts_in[:, :, row], dim=[0, 1])
        A_pp = acts[i_center + 1 :, i_center + 1 :]
        # A_mm = acts.flip(dims=(0, 1))[i_center + 1 :, i_center + 1 :]
        A_pm = acts.flip(dims=(1,))[i_center + 1 :, i_center + 1 :]
        # A_mp = acts.flip(dims=(0,))[i_center + 1 :, i_center + 1 :]
        A_0p = acts[i_center, i_center + 1 :]
        A_p0 = acts[i_center + 1 :, i_center]
        # A_0m = acts.flip(dims=(1,))[i_center, i_center + 1 :]
        # A_m0 = acts.flip(dims=(0,))[i_center + 1 :, i_center]
        A_00 = acts[i_center, i_center]

        # Make an array like this:
        # A00 A0p
        # Ap0 App
        mag_array = torch.zeros((i_center + 1, i_center + 1))
        mag_array[0, 0] = A_00.abs()
        mag_array[0, 1:] = A_0p.abs()
        mag_array[1:, 0] = A_p0.abs()
        mag_array[1:, 1:] = A_pp.abs() + A_pm.abs()
        mag_array2 = A_pp.abs() - A_pm.abs()

        phase_array = torch.zeros((i_center + 1, i_center + 1))
        phase_array[0, 0] = 0
        phase_array[0, 1:] = A_0p.angle()  # angle = atan2(im, re)
        phase_array[1:, 0] = A_p0.angle()
        phase_array[1:, 1:] = (A_pp.angle() + A_pm.angle()) / 2
        phase_array2 = (A_pp.angle() - A_pm.angle()) / 2
        vmax = max(mag_array.max(), mag_array2.max())
        axes[row, 0].imshow(
            colorize(mag_array, phase_array, vmax=vmax),
            extent=[0, i_center, 0, i_center],
            origin="lower",
        )

        axes[row, 1].imshow(
            colorize(mag_array2, phase_array2, vmax=vmax),
            extent=[1, i_center, 1, i_center],
            origin="lower",
        )
        axes[row, 0].set_ylabel(f"k_y")
        axes[row, 0].text(0.5, 0.5, f"Max val: {mag_array.max():.2e}")
        axes[row, 1].text(0.5, 0.5, f"Max val: {mag_array2.max():.2e}")
        axes[row, 0].grid(color="grey", linestyle="--", alpha=0.3)
        axes[row, 0].set_xticks(np.arange(i_center + 2))
        axes[row, 0].set_xticklabels(
            [f"cos{k:.0f}" for k in [0, *113 * freqs_plus]] + [" "], rotation=-90, fontsize=6
        )
        axes[row, 0].set_yticks(np.arange(i_center + 2))
        axes[row, 1].set_xticks(np.arange(1, i_center + 2))
        axes[row, 1].set_yticks(np.arange(1, i_center + 2))

        axes[row, 1].grid(color="grey", linestyle="--", alpha=0.3)
    axes[-1, 0].set_xlabel(f"k_x")
    axes[-1, 1].set_xlabel(f"k_x")
    axes[0, 0].set_title("cos cos")
    axes[0, 1].set_title("sin sin")
