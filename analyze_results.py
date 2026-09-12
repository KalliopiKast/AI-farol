"""
analyze_results.py
Post-hoc statistical analysis of results/raw_all_rounds.csv, produced by
run_experiments.py. Addresses:

  - R1.6 / R2.2 (Reviewers 1 & 2): confidence intervals and formal
    statistical comparisons, not just means and standard deviations.
    Computes, per (agent_type, bar_type, rounds) cell:
      * a normal-theory 95% CI (mean +/- t * s/sqrt(n))
      * a percentile bootstrap 95% CI (10,000 resamples), which does not
        assume normality -- used as the primary CI for any cell whose
        coefficient of variation exceeds 1.0 (the threshold the paper
        itself already uses to flag likely-skewed/multimodal cells).
    Also runs Welch's two-sample t-test (unequal variances, does not
    assume equal sample sizes or variances) between named pairs of
    cells for the headline comparisons the paper's Section 5 discussion
    relies on (see HEADLINE_COMPARISONS below).

  - R1.7 (Reviewer 1): a single comparable convergence/oscillation
    diagnostic per pairing, rather than only the qualitative
    coefficient-of-variation-vs-horizon plot. Fits an OLS line to
    (log10(rounds), CV) across the three horizons and reports the slope;
    a negative slope indicates dispersion shrinking as the horizon
    grows (consistent with settling toward a stable outcome), a
    positive slope indicates growing cross-seed dispersion (consistent
    with the persistent-adaptation / limit-cycle regime discussed in
    Sec. 3.3 and 5.5).

Usage:
    python analyze_results.py
Reads results/raw_all_rounds.csv, writes:
    results/ci_summary.csv
    results/welch_tests.csv
    results/cv_slopes.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

IN_PATH = os.path.join("results", "raw_all_rounds.csv")
OUT_DIR = "results"

METRICS = ["attendance", "price", "profit", "welfare"]

# Headline pairwise comparisons for the REVISED draft's Section 5 text.
# NOTE: rankings were rechecked against the real 100-seed run. no_regret x
# q_learning-bar is #1 in BOTH profit and welfare. heuristic x passive and
# heuristic x gp are close #2/#3 in profit (order can be sensitive to seed
# count -- re-verify against your own run before citing a specific rank).
# bayesian x regression remains the sharpest profit/welfare divergence
# pairing (low profit rank, #2 welfare).
# Each entry: (metric, rounds, (agent_a, bar_a), (agent_b, bar_b), description)
HEADLINE_COMPARISONS = [
    ("profit", 1000, ("no_regret", "q_learning"), ("heuristic", "passive"),
     "highest profit in the grid (no_regret x q_learning-bar) vs. second-highest (heuristic x passive)"),
    ("welfare", 1000, ("no_regret", "q_learning"), ("bayesian", "regression"),
     "highest welfare in the grid (no_regret x q_learning-bar) vs. second-highest (bayesian x regression)"),
    ("profit", 1000, ("heuristic", "gp"), ("bayesian", "regression"),
     "heuristic x gp (near the top of the profit ranking) vs. bayesian x regression (near the bottom of profit but #2 welfare) -- sharpest profit/welfare divergence in the grid"),
    ("welfare", 1000, ("bayesian", "regression"), ("heuristic", "gp"),
     "same pairing, welfare direction"),
    ("attendance", 1000, ("no_regret", "passive"), ("q_learning", "passive"),
     "no_regret vs. q_learning attendance collapse under a passive bar"),
    ("attendance", 1000, ("no_regret", "q_learning"), ("q_learning", "q_learning"),
     "no_regret vs. q_learning attendance recovery under the q_learning bar"),
]


def bootstrap_ci(values, n_boot=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def normal_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    mean = values.mean()
    sem = values.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return (float(mean - tcrit * sem), float(mean + tcrit * sem))


def compute_ci_summary(raw_df):
    rows = []
    for (agent_type, bar_type, rounds), g in raw_df.groupby(["agent_type", "bar_type", "rounds"]):
        for metric in METRICS:
            vals = g[metric].values
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            cv = abs(std / mean) if mean != 0 else np.inf
            n_lo, n_hi = normal_ci(vals)
            b_lo, b_hi = bootstrap_ci(vals)
            rows.append({
                "agent_type": agent_type, "bar_type": bar_type, "rounds": rounds,
                "metric": metric, "n_seeds": len(vals), "mean": mean, "std": std,
                "coefficient_of_variation": cv,
                "normal_ci_lo": n_lo, "normal_ci_hi": n_hi,
                "bootstrap_ci_lo": b_lo, "bootstrap_ci_hi": b_hi,
                "flagged_high_dispersion": cv > 1.0,
            })
    return pd.DataFrame(rows)


def run_welch_tests(raw_df):
    rows = []
    for metric, rounds, (a_agent, a_bar), (b_agent, b_bar), desc in HEADLINE_COMPARISONS:
        a_vals = raw_df[(raw_df.agent_type == a_agent) & (raw_df.bar_type == a_bar)
                         & (raw_df.rounds == rounds)][metric].values
        b_vals = raw_df[(raw_df.agent_type == b_agent) & (raw_df.bar_type == b_bar)
                         & (raw_df.rounds == rounds)][metric].values
        if len(a_vals) < 2 or len(b_vals) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        rows.append({
            "description": desc, "metric": metric, "rounds": rounds,
            "cell_a": f"{a_agent}/{a_bar}", "mean_a": float(np.mean(a_vals)),
            "cell_b": f"{b_agent}/{b_bar}", "mean_b": float(np.mean(b_vals)),
            "welch_t": float(t_stat), "p_value": float(p_val),
            "significant_at_0.05": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows)


def compute_cv_slopes(raw_df):
    """OLS slope of CV against log10(rounds) per (agent_type, bar_type, metric)."""
    rows = []
    for (agent_type, bar_type), g in raw_df.groupby(["agent_type", "bar_type"]):
        for metric in METRICS:
            cvs, logs = [], []
            for rounds, gg in g.groupby("rounds"):
                vals = gg[metric].values
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
                if mean == 0 or np.isnan(std):
                    continue
                cvs.append(abs(std / mean))
                logs.append(np.log10(rounds))
            if len(cvs) < 2:
                continue
            slope, intercept, r, p, se = stats.linregress(logs, cvs)
            rows.append({
                "agent_type": agent_type, "bar_type": bar_type, "metric": metric,
                "cv_vs_log10_rounds_slope": float(slope), "r_value": float(r),
                "p_value": float(p),
                "interpretation": ("growing dispersion (persistent-adaptation signature)"
                                    if slope > 0 else "shrinking/flat dispersion (settling)"),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(
            f"{IN_PATH} not found -- run run_experiments.py first to generate raw per-seed data.")
    raw_df = pd.read_csv(IN_PATH)

    ci_df = compute_ci_summary(raw_df)
    ci_df.to_csv(os.path.join(OUT_DIR, "ci_summary.csv"), index=False)
    print(f"Wrote {os.path.join(OUT_DIR, 'ci_summary.csv')} "
          f"({(ci_df.flagged_high_dispersion).sum()} cell-metrics flagged CV>1, "
          f"bootstrap CI recommended for those)")

    welch_df = run_welch_tests(raw_df)
    welch_df.to_csv(os.path.join(OUT_DIR, "welch_tests.csv"), index=False)
    print(f"Wrote {os.path.join(OUT_DIR, 'welch_tests.csv')}")
    for _, row in welch_df.iterrows():
        sig = "SIGNIFICANT" if row["significant_at_0.05"] else "not significant"
        print(f"  {row['description']}: p={row['p_value']:.4f} ({sig})")

    slope_df = compute_cv_slopes(raw_df)
    slope_df.to_csv(os.path.join(OUT_DIR, "cv_slopes.csv"), index=False)
    print(f"Wrote {os.path.join(OUT_DIR, 'cv_slopes.csv')}")