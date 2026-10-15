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
    present: np.ndarray | None = None,
    components: tuple[int, int] = (0, 1),
) -> tuple[plt.Figure, np.ndarray]:
    """Project mean activations onto principal components and plot.

    Args:
        mean_activations: Array of shape (num_concepts, hidden_dim).
        labels: List of concept label strings for annotation.
        n_components: Number of PCA components to compute.
        title: Plot title.
        ax: Matplotlib axes to plot on. Creates new figure if None.
        present: Optional boolean mask of shape (num_concepts,). Absent
            concepts are excluded from the fit; their zero rows would otherwise
            sit far from every real activation and dominate the components.
        components: Zero-indexed pair of components to draw. Fig. 3 needs
            (2, 3) for PCs 3-4, where the in-context structure lives.

    Returns:
        Tuple of (figure, projected_data of shape (num_present, n_components)).
        The whole projection comes back, not just the pair that was drawn.
    """
    if present is not None:
        mean_activations = mean_activations[present]
        labels = [l for l, keep in zip(labels, present) if keep]

    n_components = min(max(n_components, max(components) + 1),
                       *mean_activations.shape)
    if max(components) >= n_components:
        raise ValueError(
            f"Cannot plot components {components}: only {n_components} are "
            f"available from {mean_activations.shape[0]} points."
        )

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(mean_activations)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    cx, cy = components
    ax.scatter(projected[:, cx], projected[:, cy], s=60, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (projected[i, cx], projected[i, cy]),
                    fontsize=8, ha="center", va="bottom")

    ax.set_xlabel(f"PC{cx + 1} ({pca.explained_variance_ratio_[cx]:.1%})")
    ax.set_ylabel(f"PC{cy + 1} ({pca.explained_variance_ratio_[cy]:.1%})")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig, projected
