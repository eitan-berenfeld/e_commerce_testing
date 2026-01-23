import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import matplotlib.pyplot as plt

def welch_ttest(mean_c: float, mean_t: float, var_c: float, var_t: float, n_c: float, n_t: float) -> tuple:
    """
    Perform Welch's t-test for two independent samples with unequal variances.
    
    This function implements Welch's t-test, which is robust to unequal variances
    and sample sizes. It's appropriate when the assumption of equal variances
    cannot be made.
    
    Args:
        mean_c: Mean of control group
        mean_t: Mean of treatment group
        var_c: Variance of control group
        var_t: Variance of treatment group
        n_c: Sample size of control group
        n_t: Sample size of treatment group
    
    Returns:
        Tuple of (t_statistic, p_value, ci_lower, ci_upper, standard_error, degrees_of_freedom)
        Returns NaN values and p=1.0 for invalid inputs (e.g., n < 2, invalid SE)
    """
    se = np.sqrt(var_c/n_c + var_t/n_t)
    if (n_c < 2) or (n_t < 2) or (not np.isfinite(se)) or (se <= 0):
        return np.nan, 1.0, np.nan, np.nan, se, np.nan

    t_stat = (mean_t - mean_c) / se
    v1, v2 = var_c/n_c, var_t/n_t
    denom = (v1**2/(n_c-1)) + (v2**2/(n_t-1))
    df = (v1 + v2)**2 / denom if denom > 0 else np.nan
    if not np.isfinite(df) or (df <= 0) or (not np.isfinite(t_stat)):
        return t_stat, 1.0, np.nan, np.nan, se, df

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    q = stats.t.ppf(0.975, df) * se
    eff = mean_t - mean_c
    return t_stat, float(p_value), eff - q, eff + q, se, df

def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the latest snapshot for each experiment-variant-metric combination.
    
    For each unique combination of experiment_id, variant_id, and metric_id,
    returns the row with the maximum time_since_start value.
    
    Args:
        df: DataFrame with columns including 'time_since_start', 'experiment_id',
            'variant_id', and 'metric_id'
    
    Returns:
        DataFrame with one row per experiment-variant-metric combination
        (the latest time point for each)
    """
    return (
        df.sort_values('time_since_start')
        .groupby(['experiment_id', 'variant_id', 'metric_id'], as_index=False)
        .last()
    )

def _welch_results(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Welch's t-test results for all rows in the input DataFrame.
    
    Applies welch_ttest() to each row and aggregates results into a new DataFrame
    with statistical test outputs including p-values, confidence intervals, and
    effect sizes.
    
    Args:
        df_slice: DataFrame with columns 'mean_c', 'mean_t', 'variance_c',
                 'variance_t', 'count_c', 'count_t', 'experiment_id', 'variant_id',
                 'metric_id', and optionally 'time_since_start'
    
    Returns:
        DataFrame with columns: experiment_id, variant_id, metric_id,
        time_since_start, n_c, n_t, mean_c, mean_t, effect, se, df,
        t_statistic, p_value, ci_lower, ci_upper
    """
    out = []
    for r in df_slice.itertuples(index=False):
        t_stat, p_val, ci_l, ci_u, se, dfree = welch_ttest(
            r.mean_c, r.mean_t, r.variance_c, r.variance_t, r.count_c, r.count_t
        )
        out.append({
            'experiment_id': r.experiment_id,
            'variant_id': r.variant_id,
            'metric_id': r.metric_id,
            'time_since_start': getattr(r, 'time_since_start', np.nan),
            'n_c': r.count_c,
            'n_t': r.count_t,
            'mean_c': r.mean_c,
            'mean_t': r.mean_t,
            'effect': r.mean_t - r.mean_c,
            'se': se,
            'df': dfree,
            't_statistic': t_stat,
            'p_value': p_val,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
        })
    return pd.DataFrame(out)

def analyze_experiments(data_path: str = None, alpha: float = 0.05) -> tuple:
    """
    Analyze experiment data using Welch's t-test and multiple testing corrections.
    
    Loads experiment data, computes Welch's t-test for each experiment-variant-metric
    combination at the latest time point, and applies both Bonferroni and
    Benjamini-Hochberg (BH) multiple testing corrections.
    
    Args:
        data_path: Path to CSV file with experiment data. If None, uses default
                  path relative to project root.
        alpha: Significance level for multiple testing corrections (default: 0.05)
    
    Returns:
        Tuple of (raw_dataframe, results_dataframe) where results_dataframe contains:
        - Original columns plus statistical test results
        - p_value: Raw p-value from Welch's t-test
        - p_value_bonferroni: Bonferroni-adjusted p-value (FWER control)
        - p_value_bh: Benjamini-Hochberg adjusted p-value (FDR control)
        - significant_bonferroni: Boolean indicating Bonferroni significance
        - significant_bh: Boolean indicating BH significance
        - neglog10_p, neglog10_q: Negative log10 of p-values for visualization
    """
    if data_path is None:
        root = Path(__file__).resolve().parents[1]
        data_path = str(root / 'data' / 'asos_digital_experiments_dataset.csv')
    df = pd.read_csv(data_path)
    latest = _latest_snapshot(df)
    results_df = _welch_results(latest)

    results_df['p_value'] = results_df['p_value'].fillna(1.0).clip(0.0, 1.0)
    p = results_df['p_value'].to_numpy()
    bonf_reject, bonf_p, _, _ = multipletests(p, alpha=alpha, method='bonferroni')
    bh_reject, bh_p, _, _ = multipletests(p, alpha=alpha, method='fdr_bh')

    results_df['p_value_bonferroni'] = bonf_p
    results_df['p_value_bh'] = bh_p
    results_df['significant_bonferroni'] = bonf_reject
    results_df['significant_bh'] = bh_reject
    results_df['neglog10_p'] = -np.log10(np.clip(results_df['p_value'], 1e-300, 1.0))
    results_df['neglog10_q'] = -np.log10(np.clip(results_df['p_value_bh'], 1e-300, 1.0))

    return df, results_df

def _save_fig(path: Path) -> None:
    """
    Save the current matplotlib figure to a file path.
    
    Creates parent directories if needed, applies tight layout, saves with
    high DPI, and closes the figure to free memory.
    
    Args:
        path: Path object where the figure should be saved
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()

def create_story_plots(raw_df: pd.DataFrame, latest_results: pd.DataFrame, out_dir: Path, alpha: float = 0.05, top_k: int = 6) -> None:
    """
    Generate a comprehensive set of visualization plots for experiment analysis.
    
    Creates 6 plots that tell the story of the experiment results:
    1. P-value distributions (raw vs BH-adjusted)
    2. BH decision boundary curve
    3. Volcano plot (effect vs significance)
    4. Discoveries by metric (BH vs Bonferroni comparison)
    5. Top experiments by number of discoveries
    6. Top discoveries over time (temporal stability)
    
    Args:
        raw_df: Full DataFrame with all time points for temporal analysis
        latest_results: DataFrame with latest snapshot results from analyze_experiments()
        out_dir: Directory path where plots will be saved
        alpha: Significance level used for corrections (default: 0.05)
        top_k: Number of top discoveries to show in temporal plot (default: 6)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_results = latest_results.copy()
    latest_results['p_value'] = latest_results['p_value'].fillna(1.0).clip(0.0, 1.0)
    latest_results['p_value_bh'] = latest_results['p_value_bh'].fillna(1.0).clip(0.0, 1.0)
    latest_results['p_value_bonferroni'] = latest_results['p_value_bonferroni'].fillna(1.0).clip(0.0, 1.0)

    # 1) p-value landscape (raw vs BH)
    fig = plt.figure(figsize=(10, 3.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(latest_results['p_value'], bins=40, color='#4C78A8', alpha=0.9)
    ax1.axvline(alpha, color='black', lw=1)
    ax1.set_title('Raw p-values (latest snapshot)')
    ax1.set_xlabel('p')
    ax1.set_ylabel('count')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(latest_results['p_value_bh'], bins=40, color='#F58518', alpha=0.9)
    ax2.axvline(alpha, color='black', lw=1)
    ax2.set_title('BH-adjusted p-values (q-values)')
    ax2.set_xlabel('q')
    _save_fig(out_dir / '01_pvalue_distributions.png')

    # 2) BH curve (sorted p vs BH critical line)
    p = np.sort(latest_results['p_value'].to_numpy())
    m = len(p)
    crit = alpha * (np.arange(1, m + 1) / m)
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(crit, p, lw=1.5)
    plt.plot([0, alpha], [0, alpha], lw=1, color='gray', alpha=0.6)
    plt.xlim(0, alpha)
    plt.ylim(0, alpha)
    plt.title('BH decision boundary (zoomed to alpha)')
    plt.xlabel('BH critical value')
    plt.ylabel('sorted raw p-value')
    _save_fig(out_dir / '02_bh_curve.png')

    # 3) Volcano-ish: effect vs -log10(q)
    plt.figure(figsize=(6.2, 4.0))
    x = latest_results['effect'].to_numpy()
    y = latest_results['neglog10_q'].to_numpy()
    sig = latest_results['significant_bh'].to_numpy().astype(bool)
    plt.scatter(x[~sig], y[~sig], s=10, alpha=0.35, color='#9E9E9E')
    plt.scatter(x[sig], y[sig], s=14, alpha=0.85, color='#D62728')
    plt.title('Effect vs FDR significance (latest snapshot)')
    plt.xlabel('effect (mean_t - mean_c)')
    plt.ylabel(r'$-\log_{10}(q)$')
    _save_fig(out_dir / '03_effect_vs_fdr.png')

    # 4) Discoveries by metric (BH vs Bonferroni)
    by_metric = (
        latest_results
        .groupby('metric_id', as_index=False)
        .agg(bh=('significant_bh', 'sum'), bonf=('significant_bonferroni', 'sum'), tests=('p_value', 'size'))
        .sort_values('metric_id')
    )
    x = np.arange(len(by_metric))
    w = 0.38
    plt.figure(figsize=(7.2, 3.8))
    plt.bar(x - w/2, by_metric['bh'], width=w, label='BH (FDR)', color='#D62728')
    plt.bar(x + w/2, by_metric['bonf'], width=w, label='Bonferroni (FWER)', color='#1F77B4')
    plt.xticks(x, by_metric['metric_id'].astype(str))
    plt.title('Significant results by metric (latest snapshot)')
    plt.xlabel('metric_id')
    plt.ylabel('discoveries')
    plt.legend(frameon=False)
    _save_fig(out_dir / '04_discoveries_by_metric.png')

    # 5) Where the signal lives (experiments with most BH discoveries)
    top_exp = (
        latest_results.groupby('experiment_id', as_index=False)
        .agg(bh=('significant_bh', 'sum'))
        .sort_values('bh', ascending=False)
        .head(15)
    )
    plt.figure(figsize=(7.2, 4.6))
    plt.barh(top_exp['experiment_id'].astype(str)[::-1], top_exp['bh'][::-1], color='#4C78A8')
    plt.title('Top experiments by BH discoveries')
    plt.xlabel('BH discoveries (count)')
    _save_fig(out_dir / '05_top_experiments.png')

    # 6) “Do the winners hold up over time?” (top_k BH signals)
    winners = (
        latest_results[latest_results['significant_bh']]
        .sort_values('p_value_bh')
        .head(top_k)[['experiment_id', 'variant_id', 'metric_id']]
        .drop_duplicates()
    )
    if len(winners):
        dfw = raw_df.merge(winners, on=['experiment_id', 'variant_id', 'metric_id'], how='inner')
        dfw = dfw.sort_values('time_since_start')
        plt.figure(figsize=(8.6, 4.8))
        for key, g in dfw.groupby(['experiment_id', 'variant_id', 'metric_id']):
            eff = (g['mean_t'] - g['mean_c']).to_numpy()
            se = np.sqrt(g['variance_c']/g['count_c'] + g['variance_t']/g['count_t']).to_numpy()
            plt.plot(g['time_since_start'], eff, lw=1.6, label=f'{key[0]} v{key[1]} m{key[2]}')
            plt.fill_between(g['time_since_start'], eff - 1.96*se, eff + 1.96*se, alpha=0.12)
        plt.axhline(0, color='black', lw=1, alpha=0.6)
        plt.title('Top BH discoveries over time (effect ± 1.96·SE)')
        plt.xlabel('time_since_start')
        plt.ylabel('effect (mean_t - mean_c)')
        plt.legend(frameon=False, fontsize=8, ncol=1, loc='best')
        _save_fig(out_dir / '06_top_discoveries_over_time.png')

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data' / 'asos_digital_experiments_dataset.csv'
    out_dir = root / 'analysis' / 'outputs'

    raw_df, latest_results = analyze_experiments(str(data_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_results.to_csv(out_dir / 'welch_latest_multiple_testing.csv', index=False)
    create_story_plots(raw_df, latest_results, out_dir)
