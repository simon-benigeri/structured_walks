"""Linear probes for walk-type classification (Experiment 2)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_walk_type_probe(
    activations: np.ndarray,
    walk_labels: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """Train a linear probe to classify walk type from activations.

    Args:
        activations: Array of shape (num_samples, hidden_dim).
        walk_labels: Integer array of shape (num_samples,) indicating walk type.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with keys:
            'mean_accuracy': mean cross-validated accuracy
            'std_accuracy': std of cross-validated accuracy
            'model': the fitted LogisticRegression model
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, activations, walk_labels, cv=cv_folds)

    clf.fit(activations, walk_labels)

    return {
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "model": clf,
    }
