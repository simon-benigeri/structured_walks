"""PCA visualization of concept representations."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_visualization(
    mean_activations: np.ndarray,
    labels: list[str],
    n_components: int = 2,
    title: str = "PCA of Concept Representations",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Project mean activations onto principal components and plot.

    Args:
        mean_activations: Array of shape (num_concepts, hidden_dim).
        labels: List of concept label strings for annotation.
        n_components: Number of PCA components to compute.
        title: Plot title.
        ax: Matplotlib axes to plot on. Creates new figure if None.

    Returns:
        Tuple of (figure, projected_data of shape (num_concepts, n_components)).
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(mean_activations)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    ax.scatter(projected[:, 0], projected[:, 1], s=60, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (projected[i, 0], projected[i, 1]),
                    fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, projected
