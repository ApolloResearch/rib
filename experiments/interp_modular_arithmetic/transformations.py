import numpy as np
import torch
from sklearn.decomposition import PCA

torch.set_grad_enabled(False)


def fft2(acts):
    """Compute the 2D FFT of the activations over the first two dimensions, assumed to be x and y"""
    return torch.fft.fft2(acts, dim=(0, 1))


def pca_activations(acts):
    """Compute the PCA over the last dimension of the activations."""
    acts_pca = acts.reshape(-1, acts.shape[-1])
    pca = PCA(n_components=acts_pca.shape[-1])
    acts_transformed = pca.fit_transform(acts_pca)
    acts_transformed = acts_transformed.reshape(acts.shape[0], acts.shape[1], -1)
    print(
        "Explained variance:",
        "".join([f"{x:.2f} " if x > 1e-5 else "." for x in pca.explained_variance_ratio_]),
    )
    acts_transformed = torch.tensor(acts_transformed)
    return acts_transformed


def svd_activations(acts):
    """Compute the SVD over the last dimension of the activations."""
    acts_flat = acts.reshape(-1, acts.shape[-1])
    U, S, Vt = np.linalg.svd(acts_flat, full_matrices=False)
    acts_transformed = np.dot(acts_flat, Vt.T)
    acts_transformed = acts_transformed.reshape(acts.shape[0], acts.shape[1], -1)
    explained_variance_ratio = S**2 / (S**2).sum()
    print(
        "Explained variance:",
        "".join([f"{x:.2f} " if x > 1e-5 else "." for x in explained_variance_ratio]),
    )
    acts_transformed = torch.tensor(acts_transformed)
    return acts_transformed
