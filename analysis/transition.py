"""Phase transition detection via piecewise linear fitting."""

import numpy as np


def find_transition_point(
    context_lengths: np.ndarray,
    accuracies: np.ndarray,
) -> dict:
    """Find the transition point in an accuracy-vs-context curve.

    Fits a piecewise linear function (two segments) and returns
    the breakpoint where the slope changes.

    Args:
        context_lengths: Array of context lengths (x-axis).
        accuracies: Array of accuracy values (y-axis), same length.

    Returns:
        Dict with keys:
            'transition_point': context length at the breakpoint
            'slow_slope': slope of the first segment
            'fast_slope': slope of the second segment
            'residual': sum of squared residuals of the fit
            'shape': 'slow_then_fast' when the second segment is steeper -- the
                phase transition the paper describes -- or 'fast_then_slow'
                when the curve is merely saturating
            'is_transition': True only for 'slow_then_fast'

    The breakpoint minimizes residual, so this always returns a value even for
    a smoothly saturating curve. Check 'is_transition' before calling the
    result a phase transition.
    """
    log_lengths = np.log(context_lengths)
    n = len(log_lengths)
    best_residual = np.inf
    best_breakpoint = None

    for k in range(2, n - 2):
        # Fit two separate lines
        x1, y1 = log_lengths[:k], accuracies[:k]
        x2, y2 = log_lengths[k:], accuracies[k:]

        slope1, intercept1 = np.polyfit(x1, y1, 1)
        slope2, intercept2 = np.polyfit(x2, y2, 1)

        pred1 = slope1 * x1 + intercept1
        pred2 = slope2 * x2 + intercept2

        residual = np.sum((y1 - pred1) ** 2) + np.sum((y2 - pred2) ** 2)

        if residual < best_residual:
            best_residual = residual
            best_breakpoint = k
            best_slopes = (slope1, slope2)

    slow, fast = best_slopes
    shape = "slow_then_fast" if fast > slow else "fast_then_slow"

    return {
        "transition_point": context_lengths[best_breakpoint],
        "slow_slope": slow,
        "fast_slope": fast,
        "residual": best_residual,
        "shape": shape,
        "is_transition": shape == "slow_then_fast",
    }
