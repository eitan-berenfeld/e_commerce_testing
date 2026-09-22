"""
Simulates what happens to false discoveries over time when a team runs
A/B tests at a fixed weekly cadence without correcting for repeated testing,
versus applying Benjamini-Hochberg (BH) correction on each week's batch.

Motivation: the ASOS analysis (multiple_testing_correction.py) shows the
one-time cost of skipping correction on a single batch of 396 tests. This
script asks the more operational question a testing team actually faces:
if we keep running tests every week at our normal pace, how fast does the
uncorrected false-discovery problem compound, and does correcting each
week's batch actually fix it?

TESTS_PER_WEEK below is a placeholder — set it to the team's real weekly
test volume before treating these numbers as anything but illustrative.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# --- placeholder cadence; replace with the team's actual weekly test volume ---
TESTS_PER_WEEK = 12
WEEKS = 26
TRUE_EFFECT_RATE = 0.15   # rough share of tests with a genuine effect, based on
                          # this team's own discovery rate order of magnitude
ALPHA = 0.05
N_SIMULATIONS = 2000
RNG_SEED = 42


def simulate_week_pvalues(n_tests: int, true_effect_rate: float, rng: np.random.Generator) -> tuple:
    """
    Draw one week's worth of p-values under a mixture of true nulls and true effects.

    True nulls are drawn Uniform(0, 1), matching the null distribution of a
    valid test statistic. True effects are drawn from a Beta(0.5, 8) distribution,
    which concentrates mass near zero to represent tests that are genuinely likely
    to come back significant.

    Args:
        n_tests: number of tests run this week
        true_effect_rate: probability that any given test has a genuine effect
        rng: numpy random generator for reproducibility

    Returns:
        Tuple of (p_values, is_true_null) arrays, each of length n_tests
    """
    is_true_null = rng.random(n_tests) >= true_effect_rate
    p_values = np.where(
        is_true_null,
        rng.uniform(0.0, 1.0, n_tests),
        rng.beta(0.5, 8.0, n_tests),
    )
    return p_values, is_true_null


def run_simulation(
    tests_per_week: int = TESTS_PER_WEEK,
    weeks: int = WEEKS,
    true_effect_rate: float = TRUE_EFFECT_RATE,
    alpha: float = ALPHA,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Run the Monte Carlo comparison of naive (uncorrected) vs BH-corrected testing.

    For each simulated run, generates `weeks` batches of `tests_per_week` tests,
    scores each week's discoveries under (a) a flat alpha threshold with no
    correction and (b) BH correction applied within that week's batch.

    Two distinct quantities are tracked, deliberately kept separate because they
    answer different questions:

    - naive_fdr / bh_fdr: the textbook per-batch FDR, Q = V/max(R, 1) computed
      fresh each week and averaged across simulations. This is what BH actually
      controls, and it should hover near `alpha` for the BH column regardless
      of how many weeks have passed.
    - prob_any_false_naive / prob_any_false_bh: the probability that at least
      one false discovery has occurred *cumulatively* by that week. Unlike the
      per-batch FDR, this rises over time even when each individual batch is
      well-controlled, because more batches means more chances for a false
      discovery to have slipped through somewhere. This is the "risk compounds
      the longer you keep testing" quantity, and it's the one that actually
      climbs.

    Returns:
        DataFrame with one row per week, containing: week, cumulative_tests,
        naive_fdr, bh_fdr, cumulative_naive_false, cumulative_bh_false,
        prob_any_false_naive, prob_any_false_bh
    """
    rng = np.random.default_rng(seed)
    weekly_records = []

    for _ in range(n_simulations):
        cumulative_naive_false = 0
        cumulative_bh_false = 0

        for week in range(1, weeks + 1):
            p_values, is_true_null = simulate_week_pvalues(tests_per_week, true_effect_rate, rng)

            naive_reject = p_values < alpha
            naive_false_this_week = int((naive_reject & is_true_null).sum())
            naive_disc_this_week = int(naive_reject.sum())
            q_naive = naive_false_this_week / naive_disc_this_week if naive_disc_this_week > 0 else 0.0

            bh_reject, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
            bh_false_this_week = int((bh_reject & is_true_null).sum())
            bh_disc_this_week = int(bh_reject.sum())
            q_bh = bh_false_this_week / bh_disc_this_week if bh_disc_this_week > 0 else 0.0

            cumulative_naive_false += naive_false_this_week
            cumulative_bh_false += bh_false_this_week

            weekly_records.append({
                'week': week,
                'cumulative_tests': week * tests_per_week,
                'q_naive': q_naive,
                'q_bh': q_bh,
                'cumulative_naive_false': cumulative_naive_false,
                'cumulative_bh_false': cumulative_bh_false,
                'any_false_naive': cumulative_naive_false > 0,
                'any_false_bh': cumulative_bh_false > 0,
            })

    weekly_df = pd.DataFrame(weekly_records)
    summary = weekly_df.groupby('week', as_index=False).mean()
    summary = summary.rename(columns={
        'q_naive': 'naive_fdr',
        'q_bh': 'bh_fdr',
        'any_false_naive': 'prob_any_false_naive',
        'any_false_bh': 'prob_any_false_bh',
    })
    return summary


def plot_fdr_trajectory(summary: pd.DataFrame, out_path: Path, tests_per_week: int) -> None:
    """
    Plot cumulative false discoveries and the compounding risk of a false alarm over time.

    Left panel shows the raw count of false "wins" accumulating week over week —
    this is the quantity that visibly climbs under uncorrected testing.
    Right panel shows the probability that at least one false discovery has
    happened by that week. This rises over time for both approaches (more
    batches means more chances for something to slip through), but far more
    slowly under BH correction.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(summary['week'], summary['cumulative_naive_false'], color='#D62728', lw=2, label='Uncorrected')
    ax1.plot(summary['week'], summary['cumulative_bh_false'], color='#4C78A8', lw=2, label='BH-corrected')
    ax1.set_title(f'Cumulative false discoveries\n({tests_per_week} tests/week)')
    ax1.set_xlabel('week')
    ax1.set_ylabel('false "significant" results (expected)')
    ax1.legend(frameon=False)

    ax2.plot(summary['week'], summary['prob_any_false_naive'], color='#D62728', lw=2, label='Uncorrected')
    ax2.plot(summary['week'], summary['prob_any_false_bh'], color='#4C78A8', lw=2, label='BH-corrected')
    ax2.set_title('Risk of at least one false "win" so far')
    ax2.set_xlabel('week')
    ax2.set_ylabel('P(≥1 false discovery by this week)')
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'analysis' / 'outputs'

    summary = run_simulation()
    summary.to_csv(out_dir / 'fdr_cadence_simulation.csv', index=False)
    plot_fdr_trajectory(summary, out_dir / '07_fdr_cadence_simulation.png', TESTS_PER_WEEK)

    final_week = summary.iloc[-1]
    print(f"After {WEEKS} weeks at {TESTS_PER_WEEK} tests/week ({int(final_week['cumulative_tests'])} tests total):")
    print(f"  Per-batch FDR (what BH actually controls) — uncorrected: {final_week['naive_fdr']:.1%}, BH: {final_week['bh_fdr']:.1%}")
    print(f"  Cumulative false discoveries (expected) — uncorrected: {final_week['cumulative_naive_false']:.1f}, BH: {final_week['cumulative_bh_false']:.1f}")
    print(f"  Risk of at least one false discovery by week {WEEKS} — uncorrected: {final_week['prob_any_false_naive']:.1%}, BH: {final_week['prob_any_false_bh']:.1%}")
