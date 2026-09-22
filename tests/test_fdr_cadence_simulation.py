import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.fdr_cadence_simulation import simulate_week_pvalues, run_simulation


class TestSimulateWeekPvalues:
    """Test cases for simulate_week_pvalues function."""

    def test_output_shapes_and_ranges(self):
        """P-values should be valid probabilities, one per test, with a boolean null flag."""
        rng = np.random.default_rng(0)
        p_values, is_true_null = simulate_week_pvalues(n_tests=50, true_effect_rate=0.2, rng=rng)

        assert len(p_values) == 50
        assert len(is_true_null) == 50
        assert np.all((p_values >= 0) & (p_values <= 1))
        assert is_true_null.dtype == bool

    def test_true_effect_rate_zero_gives_all_nulls(self):
        """With true_effect_rate=0, every test should be a true null."""
        rng = np.random.default_rng(0)
        _, is_true_null = simulate_week_pvalues(n_tests=200, true_effect_rate=0.0, rng=rng)

        assert is_true_null.all()

    def test_true_effect_rate_one_gives_all_alternatives(self):
        """With true_effect_rate=1, every test should be a true alternative."""
        rng = np.random.default_rng(0)
        _, is_true_null = simulate_week_pvalues(n_tests=200, true_effect_rate=1.0, rng=rng)

        assert not is_true_null.any()


class TestRunSimulation:
    """Test cases for run_simulation function."""

    def test_output_structure(self):
        """Result should have one row per week with the expected columns, in order."""
        summary = run_simulation(tests_per_week=10, weeks=8, n_simulations=100, seed=1)

        assert len(summary) == 8
        assert list(summary['week']) == list(range(1, 9))
        for col in ['naive_fdr', 'bh_fdr', 'cumulative_naive_false', 'cumulative_bh_false', 'prob_any_false_naive', 'prob_any_false_bh']:
            assert col in summary.columns

    def test_cumulative_tests_matches_cadence(self):
        """cumulative_tests at week w should equal w * tests_per_week."""
        summary = run_simulation(tests_per_week=15, weeks=5, n_simulations=50, seed=1)

        expected = [15 * w for w in range(1, 6)]
        assert list(summary['cumulative_tests']) == expected

    def test_uncorrected_fdr_exceeds_bh_fdr(self):
        """The whole point of the comparison: BH should reduce, not just relabel, the FDR."""
        summary = run_simulation(tests_per_week=12, weeks=20, n_simulations=500, seed=1)

        final = summary.iloc[-1]
        assert final['bh_fdr'] < final['naive_fdr']

    def test_no_true_effects_bh_fdr_stays_low(self):
        """Under the global null, the per-batch BH FDR should stay near the nominal alpha."""
        summary = run_simulation(
            tests_per_week=12, weeks=20, true_effect_rate=0.0, n_simulations=1000, seed=1
        )

        final = summary.iloc[-1]
        assert final['bh_fdr'] <= 0.10  # allows Monte Carlo noise around alpha=0.05

    def test_risk_of_any_false_discovery_climbs_over_time(self):
        """The cumulative probability of >=1 false discovery should increase with more weeks."""
        summary = run_simulation(tests_per_week=12, weeks=20, n_simulations=500, seed=1)

        early, late = summary.iloc[2], summary.iloc[-1]
        assert late['prob_any_false_naive'] >= early['prob_any_false_naive']
        assert late['prob_any_false_bh'] >= early['prob_any_false_bh']
