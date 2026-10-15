"""Tests that the implemented scales match Park et al.

Cheap structural checks, so they can run on the login node before a job is
submitted -- they catch "the pool is too small for this graph" and "the weekday
arrangement is secretly the calendar order" before any GPU time is spent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis import cycle_overlap, windowed_mean
from analysis.circular import chance_level
from experiments.reproduce import context_lengths_for, parse_args
from experiments.scaling import fit_power_law
from experiments.semantic_priors import arrange_weekdays
from graphs import make_grid_graph, make_hexagonal_graph, make_ring_graph
from graphs.tokens import CONCEPT_TOKENS, WEEKDAY_TOKENS, assign_tokens

# Every graph the paper uses, and the figure it appears in.
PAPER_GRAPHS = {
    "grid4 (Figs. 1, 4a)": (make_grid_graph(4), 16),
    "grid5 (Fig. 5b)": (make_grid_graph(5), 25),
    "ring10 (Fig. 4b)": (make_ring_graph(10), 10),
    "ring50 (Fig. 5a)": (make_ring_graph(50), 50),
    "hex30 (Fig. 4c)": (make_hexagonal_graph(3, 3), 30),
}

# Fig. 8 fits the transition point across these grid side lengths.
SCALING_SIZES = [3, 4, 5, 6]


def test_every_paper_graph_has_the_expected_node_count():
    for name, (graph, expected) in PAPER_GRAPHS.items():
        assert graph.number_of_nodes() == expected, name


def test_concept_pool_covers_the_largest_graph():
    largest = max(n for _, n in PAPER_GRAPHS.values())
    assert largest == 50, "the 50-node ring should be the biggest graph"
    assert len(CONCEPT_TOKENS) >= largest, (
        f"pool has {len(CONCEPT_TOKENS)} words but the largest graph needs {largest}"
    )


def test_concept_pool_has_no_duplicates():
    assert len(set(CONCEPT_TOKENS)) == len(CONCEPT_TOKENS)


def test_labels_can_be_assigned_to_every_paper_graph():
    for name, (graph, _) in PAPER_GRAPHS.items():
        labels = assign_tokens(graph.number_of_nodes(), seed=0)
        assert len(set(labels)) == graph.number_of_nodes(), name


def test_scaling_sizes_fit_in_the_pool():
    for m in SCALING_SIZES:
        labels = assign_tokens(m * m, seed=0)
        assert len(labels) == m * m


def test_weekday_arrangement_differs_from_calendar_order():
    """Otherwise the two orderings are indistinguishable and Fig. 3 is vacuous."""
    for seed in range(20):
        labels, ring_cycle, calendar_cycle = arrange_weekdays(seed)
        assert sorted(labels) == sorted(WEEKDAY_TOKENS)
        overlap = cycle_overlap(ring_cycle, calendar_cycle)
        assert overlap <= 0.5, f"seed {seed}: overlap {overlap}"


def test_weekday_ring_is_seven_nodes():
    assert len(WEEKDAY_TOKENS) == 7
    assert make_ring_graph(len(WEEKDAY_TOKENS)).number_of_nodes() == 7


def test_power_law_fit_recovers_a_known_exponent():
    sizes = np.array(SCALING_SIZES)
    points = 12.0 * sizes.astype(float) ** 2.5
    fit = fit_power_law(sizes, points)
    assert abs(fit["alpha"] - 2.5) < 1e-6
    assert abs(fit["coefficient"] - 12.0) < 1e-6
    assert fit["r2"] > 0.999


def test_power_law_fit_reports_poor_quality_for_noise():
    sizes = np.array(SCALING_SIZES)
    points = np.array([500.0, 120.0, 900.0, 300.0])
    assert fit_power_law(sizes, points)["r2"] < 0.5


def _default_args(**overrides):
    saved = sys.argv
    try:
        sys.argv = ["reproduce.py"]
        args = parse_args()
    finally:
        sys.argv = saved
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_accuracy_is_not_averaged_over_time_by_default():
    """Sec. 4.1 defines accuracy at a single position; the batch does the
    smoothing. A time window here would blur the transition and impose a floor
    on the shortest measurable context."""
    assert _default_args().accuracy_window == 1


def test_accuracy_window_of_one_reads_the_last_position():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert windowed_mean(values, 3, 1) == 0.3
    assert windowed_mean(values, 4, 1) == 0.4


def test_mean_activation_window_follows_section_3_1():
    """N_w = 50, and shorter contexts fall back to every token."""
    args = _default_args()
    assert args.window_size == 50
    assert args.window_mode == "fixed"


def test_log_spacing_resolves_short_contexts():
    """Linear strides put almost no points where small graphs transition."""
    args = _default_args(num_steps=5000)
    log_lengths = context_lengths_for(args)
    linear_lengths = context_lengths_for(_default_args(
        num_steps=5000, context_spacing="linear", stride=100))

    assert (log_lengths <= 500).sum() > 3 * (linear_lengths <= 500).sum()
    assert log_lengths[0] <= 10
    assert log_lengths[-1] == 5000
    assert np.all(np.diff(log_lengths) > 0), "must be sorted and deduplicated"


def test_linear_spacing_still_available():
    lengths = context_lengths_for(_default_args(
        num_steps=1000, context_spacing="linear", stride=100))
    assert lengths[0] == 100 and lengths[-1] == 1000
    assert len(lengths) == 10


def test_chance_level_falls_with_graph_size():
    """The 7-node weekday ring has a high floor; the 50-node ring does not."""
    assert chance_level(7) > 0.3
    assert chance_level(50) < 0.05


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print("scale tests PASSED")
