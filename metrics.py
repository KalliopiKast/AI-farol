"""
metrics.py
Welfare and equilibrium diagnostics for the AI-Farol simulations
(Section 4 of the paper).
"""

import numpy as np
from model import satisfaction, bar_cost


def welfare_series(K_history, price_history, cfg):
    """Agent surplus, bar profit, and social welfare per period (Eq. 13-14)."""
    K = np.asarray(K_history, dtype=float)
    p = np.asarray(price_history, dtype=float)
    U = satisfaction(K, cfg)
    agent_surplus = K * (U - p)
    profit = p * K - bar_cost(K, cfg)
    welfare = K * U - bar_cost(K, cfg)
    return agent_surplus, profit, welfare


def empirical_regret(agent, realized_K, realized_price, cfg):
    """For a NoRegretAgent: compares its ACTUAL cumulative utility (already
    logged in agent.utility_history) against the best FIXED strategy in
    hindsight, replayed against the same realized (K_t, price_t) sequence.

    A near-zero, decaying average regret is the empirical signature that
    the population is approaching a coarse correlated equilibrium
    (Hart & Mas-Colell, 2000), as invoked in Sections 3.2 and 4.1.

    Returns (avg_regret, cumulative_reward_per_strategy).
    """
    strategies = agent.strategies
    n = len(realized_K)
    cum_reward_per_strategy = np.zeros(len(strategies))
    hist = []
    for t in range(n):
        K_t, price_t = realized_K[t], realized_price[t]
        u_t = satisfaction(K_t, cfg) - price_t
        for i, strat in enumerate(strategies):
            a = strat(hist)
            cum_reward_per_strategy[i] += a * u_t
        hist.append((t, K_t, price_t))

    actual_cum_reward = float(np.sum(agent.utility_history[:n]))
    best_fixed = float(cum_reward_per_strategy.max())
    avg_regret = (best_fixed - actual_cum_reward) / n if n else np.nan
    return avg_regret, cum_reward_per_strategy


def rolling_regret(agent, realized_K, realized_price, cfg, window=50):
    """Windowed average regret over time, to visualize whether regret is
    decaying (the no-regret guarantee) rather than just reporting a
    single end-of-run number.
    """
    n = len(realized_K)
    out = []
    for end in range(window, n + 1, max(1, window // 5)):
        r, _ = empirical_regret(agent, realized_K[:end], realized_price[:end], cfg)
        out.append((end, r))
    return out
