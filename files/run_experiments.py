"""
run_experiments.py
Monte Carlo experiment runner: for each {agent_type} x {bar_type}
combination, runs N_SEEDS independent simulations at each of several
round counts, and reports mean +/- standard deviation across seeds for
every tracked quantity (attendance, price, profit, social welfare).

Previously this script ran a single seed at a fixed, arbitrary round
count (300) -- that gave no sense of run-to-run variability and no way
to see how behavior changes as the learning processes are given more
time. This version fixes both: proper Monte Carlo (multiple seeds) at
several round counts (10, 100, 1000), matching the round-count folders
already used elsewhere in the project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import GameConfig
from simulation import run_simulation
from metrics import welfare_series, empirical_regret, rolling_regret
from bar import _HAS_GP

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

AGENT_TYPES = ["heuristic", "bayesian", "no_regret", "q_learning"]
BAR_TYPES = ["passive", "regression", "q_learning"] + (["gp"] if _HAS_GP else [])
ROUND_COUNTS = [10, 100, 1000]
N_SEEDS = 20


def _run_one_seed(agent_type, bar_type, rounds, seed):
    """Run a single simulation and return per-round-averaged summary stats
    for this one seed (attendance, price, profit, welfare)."""
    cfg = GameConfig(rounds=rounds, seed=seed)
    df, agents, bar = run_simulation(agent_type, bar_type, cfg=cfg, seed=seed)
    _, _, welfare = welfare_series(df["attendance"], df["price"], cfg)
    return {
        "attendance": df["attendance"].mean(),
        "price": df["price"].mean(),
        "profit": df["profit"].mean(),
        "welfare": welfare.mean(),
        "distress_pct": bar.distress_periods / rounds * 100,
    }


def monte_carlo_summary(agent_type, bar_type, rounds, n_seeds=N_SEEDS):
    """Run n_seeds independent simulations and return mean/std across
    seeds for each tracked quantity -- this IS the Monte Carlo estimate."""
    per_seed = [_run_one_seed(agent_type, bar_type, rounds, seed) for seed in range(n_seeds)]
    df = pd.DataFrame(per_seed)
    summary = {}
    for col in df.columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()
    return summary


def run_grid(round_counts=ROUND_COUNTS, n_seeds=N_SEEDS):
    """Full sweep: every agent x bar combination, at every round count,
    with Monte Carlo mean/std across seeds. Returns one long-format
    DataFrame and also writes one CSV per round count."""
    rows = []
    for rounds in round_counts:
        print(f"\n=== rounds = {rounds} (Monte Carlo over {n_seeds} seeds) ===")
        round_rows = []
        for agent_type in AGENT_TYPES:
            for bar_type in BAR_TYPES:
                summary = monte_carlo_summary(agent_type, bar_type, rounds, n_seeds)
                row = {"agent_type": agent_type, "bar_type": bar_type, "rounds": rounds}
                row.update(summary)
                rows.append(row)
                round_rows.append(row)
                print(f"  {agent_type:12s} {bar_type:12s} "
                      f"K={summary['attendance_mean']:6.2f}+/-{summary['attendance_std']:5.2f}  "
                      f"p={summary['price_mean']:6.2f}+/-{summary['price_std']:5.2f}  "
                      f"profit={summary['profit_mean']:7.2f}+/-{summary['profit_std']:6.2f}  "
                      f"welfare={summary['welfare_mean']:7.2f}+/-{summary['welfare_std']:6.2f}")
        pd.DataFrame(round_rows).to_csv(os.path.join(OUT_DIR, f"summary_{rounds}rounds.csv"), index=False)

    full_df = pd.DataFrame(rows)
    full_df.to_csv(os.path.join(OUT_DIR, "summary_all_rounds.csv"), index=False)
    return full_df


def plot_convergence(full_df, agent_type, bar_type, metric="attendance"):
    """Plot how a given metric's mean +/- std evolves across round counts,
    for one agent x bar combination -- shows convergence behavior
    directly, which a single fixed round count cannot."""
    sub = full_df[(full_df.agent_type == agent_type) & (full_df.bar_type == bar_type)]
    sub = sub.sort_values("rounds")
    plt.figure(figsize=(6, 4))
    plt.errorbar(sub["rounds"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                 marker="o", capsize=4)
    plt.xscale("log")
    plt.xlabel("Rounds (log scale)")
    plt.ylabel(metric)
    plt.title(f"{agent_type} + {bar_type}: {metric} vs. rounds")
    plt.tight_layout()
    fname = f"convergence_{agent_type}_{bar_type}_{metric}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()


def plot_combined_convergence(full_df, combos, metric="attendance"):
    """Plot the same metric's convergence for SEVERAL combos on one chart,
    one line per combo, so they can be compared directly instead of
    flipping between separate single-combo plots. This is the
    "un-split" view: one figure per metric (not one per combo x metric).
    """
    plt.figure(figsize=(7, 5))
    for agent_type, bar_type in combos:
        sub = full_df[(full_df.agent_type == agent_type) & (full_df.bar_type == bar_type)]
        sub = sub.sort_values("rounds")
        label = f"{agent_type} + {bar_type}"
        plt.errorbar(sub["rounds"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                     marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xlabel("Rounds (log scale)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. rounds, across game combinations")
    plt.legend(fontsize=9)
    plt.tight_layout()
    fname = f"combined_convergence_{metric}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()


if __name__ == "__main__":
    full_df = run_grid()

    # Convergence plots for the four illustrative combinations, across
    # all four tracked metrics.
    key_combos = [
        ("heuristic", "passive"),
        ("heuristic", "q_learning"),
        ("no_regret", "passive"),
        ("no_regret", "q_learning"),
    ]
    for agent_type, bar_type in key_combos:
        for metric in ["attendance", "price", "profit", "welfare"]:
            plot_convergence(full_df, agent_type, bar_type, metric)

    # Combined view: one plot per metric, all 4 key combos overlaid,
    # instead of 16 separate small plots.
    for metric in ["attendance", "price", "profit", "welfare"]:
        plot_combined_convergence(full_df, key_combos, metric)

    print(f"\nAll CSVs and convergence plots saved to ./{OUT_DIR}/")
