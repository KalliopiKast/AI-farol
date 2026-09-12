"""
model.py
Core configuration and shared functions for the AI-Farol two-sided game.

Implements the mathematical objects from Sections 2 and 4 of the paper:
  - satisfaction function U(K)                (congestion-sensitive, concave)
  - bar cost function C_B(K)
  - agent utility u_i^t = a_i^t * [U(K_t) - p_t]      (Eq. 5)
  - bar profit Pi_t = p_t K_t - C_B(K_t)               (Eq. 2)
  - social welfare W_total = sum U(K) - C_B(K)         (Eq. 14)
  - constraint checking: K_min <= K_t <= K_max, Pi_t >= Pi_min   (Eq. 3)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GameConfig:
    n_agents: int = 10
    k_target: int = 6      # comfortable / target attendance level
    u_max: float = 10.0     # peak satisfaction at k_target
    sigma: float = 3.0      # spread of the satisfaction curve
    c_fixed: float = 1.0    # bar fixed operating cost
    c_var: float = 0.3      # bar variable cost per attendee
    p0: float = 10.0        # baseline price
    d_max: float = 8.0      # maximum discount (price ranges in [p0-d_max, p0])
    k_min: int = 2          # minimum occupancy constraint
    k_max: int = 10         # maximum capacity constraint
    pi_min: float = -5.0    # minimum profit before "distress" / near-bankruptcy
    rounds: int = 300
    seed: int = 0
    obs_subset_size: int = 3   # |S_i^t|: size of the subset of OTHER agents each
                                # attending agent observes (Eq. 6, 11). Set to None
                                # to fall back to the aggregate-only private-history
                                # regime (agent sees true K_t on nights it attends).


def satisfaction(K, cfg: GameConfig):
    """Concave, congestion-sensitive satisfaction function U(K).
    A Gaussian bump peaking at cfg.k_target: enjoyable near the target
    occupancy, falling off (but never negative) both when too empty and
    when overcrowded.
    """
    K = np.asarray(K, dtype=float)
    return cfg.u_max * np.exp(-((K - cfg.k_target) ** 2) / (2 * cfg.sigma ** 2))


def bar_cost(K, cfg: GameConfig):
    """Attendance-dependent operating cost C_B(K_t), Eq. (2)."""
    K = np.asarray(K, dtype=float)
    return cfg.c_fixed + cfg.c_var * K


def agent_utility(attended, K, price, cfg: GameConfig):
    """u_i^t = a_i^t * [U(K_t) - p_t], Eq. (5)."""
    return attended * (satisfaction(K, cfg) - price)


def bar_profit(price, K, cfg: GameConfig):
    """Pi_t = p_t K_t - C_B(K_t), Eq. (2)."""
    return price * K - bar_cost(K, cfg)


def social_welfare(K, price, cfg: GameConfig):
    """W_total = sum_i U(K) over attendees - C_B(K); prices cancel out
    since they are pure transfers from agents to the bar, Eq. (14).
    """
    return K * satisfaction(K, cfg) - bar_cost(K, cfg)


def constraints_satisfied(K, profit, cfg: GameConfig):
    """Check the three bar constraints from Eq. (3)."""
    return (cfg.k_min <= K <= cfg.k_max) and (profit >= cfg.pi_min)
