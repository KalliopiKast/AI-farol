"""
run_experiments.py
Runs the full grid of {agent_type} x {bar_type} combinations, saves CSVs,
and produces comparison plots plus no-regret/CCE diagnostics.

This script generates the empirical results supporting the theoretical
claims of Sections 3-4, addressing the limitation flagged in Section 5:
"the model generates testable predictions ... that warrant empirical
validation" -- here validated in simulation, as a first step.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from model import GameConfig
from simulation import run_simulation
from metrics import welfare_series, empirical_regret, rolling_regret
from bar import _HAS_GP

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

AGENT_TYPES = ["heuristic", "bayesian", "no_regret", "q_learning"]
BAR_TYPES = ["passive", "regression", "q_learning"] + (["gp"] if _HAS_GP else [])


def run_grid(rounds=300, seed=0):
    cfg = GameConfig(rounds=rounds, seed=seed)
    all_results = {}
    for agent_type in AGENT_TYPES:
        for bar_type in BAR_TYPES:
            df, agents, bar = run_simulation(agent_type, bar_type, cfg=cfg, seed=seed)
            surplus, profit_chk, welfare = welfare_series(df["attendance"], df["price"], cfg)
            df["agent_surplus"] = surplus
            df["social_welfare"] = welfare

            key = f"{agent_type}__{bar_type}"
            df.to_csv(os.path.join(OUT_DIR, f"{key}.csv"), index=False)
            all_results[key] = df

            msg = (f"[{key}] mean attendance={df['attendance'].mean():.2f}, "
                   f"mean profit={df['profit'].mean():.2f}, "
                   f"distress periods={bar.distress_periods}/{cfg.rounds}")

            if agent_type == "no_regret":
                regret, _ = empirical_regret(
                    agents[0], df["attendance"].values, df["price"].values, cfg
                )
                msg += f", final avg regret (agent 0)={regret:.4f}"
                curve = rolling_regret(agents[0], df["attendance"].values,
                                        df["price"].values, cfg, window=50)
                _plot_regret_curve(curve, key)

            print(msg)
    return all_results


def _plot_regret_curve(curve, key):
    xs, ys = zip(*curve)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", ms=3)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Round")
    plt.ylabel("Average regret (agent 0)")
    plt.title(f"No-regret convergence check: {key}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"regret_{key}.png"), dpi=150)
    plt.close()


def plot_comparison(all_results, metric="attendance", title=None):
    plt.figure(figsize=(10, 5))
    for key, df in all_results.items():
        plt.plot(df["t"], df[metric], label=key, alpha=0.7, linewidth=1)
    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"comparison_{metric}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    results = run_grid(rounds=300, seed=0)
    for metric in ["attendance", "price", "profit", "social_welfare"]:
        plot_comparison(results, metric=metric)
    print(f"\nAll CSVs and plots saved to ./{OUT_DIR}/")
