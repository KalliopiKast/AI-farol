"""
simulation.py
Full two-sided dynamic game loop tying an agent population and a bar
together, implementing the exact 7-step sequence of Section 2.2:

  (1) bar observes state, chooses discount d_t via policy, sets p_t
  (2) agents observe p_t, form attendance decisions from beliefs/strategy
  (3) attendance realizes as K_t = sum_i a_i^t
  (4) attending agents receive u_i^t = a_i^t [U(K_t) - p_t]
  (5) bar receives Pi_t = p_t K_t - C_B(K_t)
  (6) attending agents observe subsets S_i^t and update beliefs
  (7) bar observes K_t, R_t and updates its policy
"""

import numpy as np
import pandas as pd

from model import GameConfig, agent_utility, bar_profit, constraints_satisfied
from agents import AGENT_REGISTRY
from bar import BAR_REGISTRY


def build_agents(agent_type, cfg, rng, **kwargs):
    cls = AGENT_REGISTRY[agent_type]
    if agent_type == "q_learning":
        price_buckets = np.linspace(cfg.p0 - cfg.d_max, cfg.p0, 5)
        return [cls(i, cfg, price_buckets=price_buckets, rng=rng, **kwargs)
                for i in range(cfg.n_agents)]
    return [cls(i, cfg, rng=rng, **kwargs) for i in range(cfg.n_agents)]


def build_bar(bar_type, cfg, rng, **kwargs):
    cls = BAR_REGISTRY[bar_type]
    return cls(cfg, rng=rng, **kwargs)


def run_simulation(agent_type="heuristic", bar_type="passive", cfg=None, seed=0,
                    agent_kwargs=None, bar_kwargs=None):
    """Runs one full simulation and returns (dataframe, agents, bar)."""
    cfg = cfg or GameConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    agents = build_agents(agent_type, cfg, rng, **(agent_kwargs or {}))
    bar = build_bar(bar_type, cfg, rng, **(bar_kwargs or {}))

    records = []
    for t in range(cfg.rounds):
        price = bar.set_price(t)                                  # (1)
        decisions = [a.decide(t, price) for a in agents]           # (2)
        K_t = int(sum(decisions))                                  # (3)
        profit_t = bar_profit(price, K_t, cfg)                     # (5), computed here for logging
        feasible = constraints_satisfied(K_t, profit_t, cfg)

        for a, d in zip(agents, decisions):                        # (4) + (6)
            a.observe(t, price, K_t, d)
        bar.observe(t, K_t, price, profit_t)                       # (7)

        mean_utility = float(np.mean([
            agent_utility(d, K_t, price, cfg) for d in decisions
        ]))

        records.append({
            "t": t,
            "price": price,
            "attendance": K_t,
            "profit": profit_t,
            "feasible": feasible,
            "mean_agent_utility": mean_utility,
        })

    df = pd.DataFrame(records)
    return df, agents, bar
