import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import analysis module
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.multiple_testing_correction import (
    welch_ttest,
    _latest_snapshot,
    _welch_results,
    analyze_experiments
)


class TestWelchTtest:
    """Test cases for welch_ttest function."""
    
    def test_basic_welch_ttest(self):
        """Test Welch's t-test with known values."""
        # Known test case: control vs treatment
        mean_c, mean_t = 10.0, 12.0
        var_c, var_t = 4.0, 5.0
        n_c, n_t = 20, 25
        
        t_stat, p_val, ci_l, ci_u, se, df = welch_ttest(
            mean_c, mean_t, var_c, var_t, n_c, n_t
        )
        
        # Check that results are finite and reasonable
        assert np.isfinite(t_stat)
        assert np.isfinite(p_val)
        assert 0 <= p_val <= 1
        assert np.isfinite(se)
        assert se > 0
        assert df > 0
        assert ci_l < ci_u
        
        # Effect should be positive (treatment > control)
        assert (ci_l + ci_u) / 2 > 0
    
    def test_welch_ttest_negative_effect(self):
        """Test with negative effect (treatment < control)."""
        mean_c, mean_t = 12.0, 10.0
        var_c, var_t = 4.0, 5.0
        n_c, n_t = 20, 25
        
        t_stat, p_val, ci_l, ci_u, se, df = welch_ttest(
            mean_c, mean_t, var_c, var_t, n_c, n_t
        )
        
        assert np.isfinite(t_stat)
        assert np.isfinite(p_val)
        assert 0 <= p_val <= 1
        # Effect should be negative
        assert (ci_l + ci_u) / 2 < 0
    
    def test_welch_ttest_small_sample(self):
        """Test edge case with small sample size."""
        mean_c, mean_t = 10.0, 12.0
        var_c, var_t = 4.0, 5.0
        n_c, n_t = 1, 2  # Too small
        
        t_stat, p_val, ci_l, ci_u, se, df = welch_ttest(
            mean_c, mean_t, var_c, var_t, n_c, n_t
        )
        
        # Should return NaN and p=1.0 for invalid inputs
        assert np.isnan(t_stat) or not np.isfinite(t_stat)
        assert p_val == 1.0
        assert np.isnan(ci_l) or not np.isfinite(ci_l)
    
    def test_welch_ttest_zero_variance(self):
        """Test edge case with zero variance."""
        mean_c, mean_t = 10.0, 12.0
        var_c, var_t = 0.0, 0.0
        n_c, n_t = 20, 25
        
        t_stat, p_val, ci_l, ci_u, se, df = welch_ttest(
            mean_c, mean_t, var_c, var_t, n_c, n_t
        )
        
        # Should handle gracefully
        assert p_val == 1.0 or not np.isfinite(p_val)
    
    def test_welch_ttest_missing_variance(self):
        """Test with NaN variance values."""
        mean_c, mean_t = 10.0, 12.0
        var_c, var_t = np.nan, 5.0
        n_c, n_t = 20, 25
        
        t_stat, p_val, ci_l, ci_u, se, df = welch_ttest(
            mean_c, mean_t, var_c, var_t, n_c, n_t
        )
        
        # Should return NaN or p=1.0
        assert p_val == 1.0 or not np.isfinite(p_val)


class TestLatestSnapshot:
    """Test cases for _latest_snapshot function."""
    
    def test_latest_snapshot_basic(self):
        """Test extracting latest snapshot for each combination."""
        df = pd.DataFrame({
            'experiment_id': ['A', 'A', 'B', 'B'],
            'variant_id': [1, 1, 2, 2],
            'metric_id': [1, 1, 1, 1],
            'time_since_start': [0, 10, 5, 15],
            'mean_c': [1, 2, 3, 4]
        })
        
        result = _latest_snapshot(df)
        
        assert len(result) == 2
        assert result['time_since_start'].iloc[0] == 10
        assert result['time_since_start'].iloc[1] == 15
    
    def test_latest_snapshot_multiple_metrics(self):
        """Test with multiple metrics per experiment."""
        df = pd.DataFrame({
            'experiment_id': ['A', 'A', 'A', 'A'],
            'variant_id': [1, 1, 1, 1],
            'metric_id': [1, 1, 2, 2],
            'time_since_start': [0, 10, 5, 15],
            'mean_c': [1, 2, 3, 4]
        })
        
        result = _latest_snapshot(df)
        
        assert len(result) == 2
        assert set(result['metric_id']) == {1, 2}


class TestWelchResults:
    """Test cases for _welch_results function."""
    
    def test_welch_results_basic(self):
        """Test computing Welch results for a DataFrame slice."""
        df_slice = pd.DataFrame({
            'experiment_id': ['A', 'B'],
            'variant_id': [1, 2],
            'metric_id': [1, 1],
            'mean_c': [10.0, 12.0],
            'mean_t': [12.0, 11.0],
            'variance_c': [4.0, 5.0],
            'variance_t': [5.0, 4.0],
            'count_c': [20, 25],
            'count_t': [25, 20]
        })
        
        result = _welch_results(df_slice)
        
        assert len(result) == 2
        assert 'p_value' in result.columns
        assert 't_statistic' in result.columns
        assert 'effect' in result.columns
        assert 'ci_lower' in result.columns
        assert 'ci_upper' in result.columns
        assert all(result['p_value'] >= 0)
        assert all(result['p_value'] <= 1)


class TestAnalyzeExperiments:
    """Test cases for analyze_experiments function."""
    
    def test_analyze_experiments_integration(self):
        """Integration test with actual data file."""
        root = Path(__file__).parent.parent
        data_path = root / 'data' / 'asos_digital_experiments_dataset.csv'
        
        if not data_path.exists():
            pytest.skip("Data file not found")
        
        raw_df, results_df = analyze_experiments(str(data_path), alpha=0.05)
        
        # Check that results are returned
        assert isinstance(raw_df, pd.DataFrame)
        assert isinstance(results_df, pd.DataFrame)
        
        # Check required columns in results
        required_cols = [
            'p_value', 'p_value_bonferroni', 'p_value_bh',
            'significant_bonferroni', 'significant_bh',
            'effect', 't_statistic'
        ]
        for col in required_cols:
            assert col in results_df.columns
        
        # Check p-values are in valid range
        assert all(results_df['p_value'] >= 0)
        assert all(results_df['p_value'] <= 1)
        assert all(results_df['p_value_bh'] >= 0)
        assert all(results_df['p_value_bh'] <= 1)
        
        # Check that BH finds at least as many as Bonferroni
        bh_count = results_df['significant_bh'].sum()
        bonf_count = results_df['significant_bonferroni'].sum()
        assert bh_count >= bonf_count
