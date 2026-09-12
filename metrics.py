"""
metrics.py
Welfare and equilibrium diagnostics for the AI-Farol simulations
(Section 4 of the paper).
"""

import copy

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
    hindsight.

    A near-zero, decaying average regret is the empirical signature that
    the population is approaching a coarse correlated equilibrium
    (Hart & Mas-Colell, 2000), as invoked in Sections 3.2 and 4.1.

    COUNTERFACTUAL CORRECTION: a candidate strategy that would have played
    a DIFFERENT action than the agent's own realized action a_i^t also
    changes total attendance that round, and hence the congestion-sensitive
    payoff U(K_t). Round t's realized attendance decomposes as
    K_t = K_t^(-i) + a_i^t, where K_t^(-i) is what every OTHER agent
    actually did (unaffected by this agent's counterfactual, since actions
    are simultaneous within a round). So swapping in a candidate action a'
    gives the exact one-round counterfactual attendance
    K_t' = K_t - a_i^t + a', which is what is scored below -- rather than
    (as in the earlier, uncorrected version of this function) reusing the
    realized K_t regardless of what the candidate strategy would have done.
    This does not attempt to propagate the counterfactual to how OTHER
    agents' beliefs or future play would have evolved under a genuinely
    different history; that would require re-simulating the whole game and
    is out of scope here (see paper Sec. 5.6 discussion / Sec. 6.3 future
    work).

    Returns (avg_regret, cumulative_reward_per_strategy).
    """
    strategies = agent.strategies
    n = min(len(realized_K), len(agent.action_history))
    cum_reward_per_strategy = np.zeros(len(strategies))
    hist = []
    for t in range(n):
        K_t, price_t = realized_K[t], realized_price[t]
        own_action_t = agent.action_history[t]
        for i, strat in enumerate(strategies):
            a = strat(hist)
            K_counterfactual = K_t - own_action_t + a
            u_t = satisfaction(K_counterfactual, cfg) - price_t
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


def empirical_regret_uncorrected(agent, realized_K, realized_price, cfg):
    """Reproduces the ORIGINAL (flawed) regret calculation from the first
    submission, for the R3.2 before/after comparison in Section 5.9 ONLY
    -- not for any other use. Reuses the realized K_t for every candidate
    strategy regardless of what that strategy would have actually done,
    silently holding congestion fixed while changing the agent's own
    action (see empirical_regret's docstring for why this understates
    regret). Computed on the SAME simulated (realized_K, realized_price,
    agent.utility_history) as empirical_regret, so the two functions'
    outputs are directly comparable on identical underlying data -- this
    isolates the effect of the counterfactual correction itself, with no
    other confound (unlike re-running old, unfixed simulation code, which
    would also change the regression bar's price-awareness, GP bar's
    windowing, etc. simultaneously).
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


def regret_before_after(agent, realized_K, realized_price, cfg):
    """Runs both the uncorrected and corrected regret calculations on the
    IDENTICAL underlying simulated data for one agent, and returns both
    so Section 5.9 can report the R3.2 correction's effect directly.
    """
    old_regret, _ = empirical_regret_uncorrected(agent, realized_K, realized_price, cfg)
    new_regret, _ = empirical_regret(agent, realized_K, realized_price, cfg)
    return {"uncorrected_regret": old_regret, "corrected_regret": new_regret}


def population_regret(agents, realized_K, realized_price, cfg):
    """Average empirical regret (see empirical_regret) across every agent
    in a no-regret population for a single run. Reporting a single agent
    from a single run (as the original figure/analysis did) is not
    representative of the population; this averages across all n_agents
    in one run. Returns (mean_regret, std_regret_across_agents)."""
    regrets = [empirical_regret(a, realized_K, realized_price, cfg)[0] for a in agents]
    return float(np.mean(regrets)), float(np.std(regrets))


def convergence_round(series, window=200, rel_tol=0.02, sustain=3, step=None):
    """Pre-registered convergence diagnostic for R3.4 (Reviewer 3): finds
    the first round at which a metric's trailing rolling mean has
    stabilized within `rel_tol` relative change, sustained for `sustain`
    consecutive checkpoints, and stays there for the remainder of the
    series actually checked. This is a diagnostic on a SINGLE run's
    round-by-round time series (distinct from the CV-vs-horizon
    cross-seed dispersion diagnostic used elsewhere for R1.7, which
    compares seeds at fixed checkpoints rather than looking within one
    run).

    Checkpoints use NON-overlapping windows by default (step=window).
    An earlier version defaulted to step=window//2 (50% overlap), which
    produced spuriously early "converged" verdicts: with half the data
    shared between consecutive checkpoints, their means are similar
    almost by construction, regardless of whether the underlying process
    has actually settled (caught by a sanity check against a pairing the
    paper's own longer-horizon data showed still declining at round
    1000 -- the overlapping-window version incorrectly reported
    convergence by round 300). Non-overlapping blocks make each
    checkpoint an independent summary of a fresh stretch of rounds, so
    agreement between them is a real signal.

    Parameters are pre-registered here (fixed before looking at results)
    rather than tuned to make any particular pairing look converged or
    not: window=200 rounds, rel_tol=2% relative change between
    consecutive non-overlapping block means, sustained for 3 consecutive
    blocks (i.e. the criterion must hold, without reverting, across
    3 full, disjoint 200-round blocks -- 600 rounds of stability).

    Returns the round index at which the criterion first becomes
    satisfied and remains satisfied through the end of the series, or
    None if the series never reaches that point within its length.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    step = step or window
    if n < window * (sustain + 1):
        return None

    checkpoints = list(range(window, n + 1, step))
    means = [float(np.mean(series[max(0, end - window):end])) for end in checkpoints]

    consec = 0
    convergence_start = None
    for i in range(1, len(means)):
        prev, curr = means[i - 1], means[i]
        denom = abs(prev) if abs(prev) > 1e-9 else 1e-9
        rel_change = abs(curr - prev) / denom
        if rel_change < rel_tol:
            consec += 1
            if consec == 1:
                convergence_start = checkpoints[i - 1]
            if consec >= sustain:
                return convergence_start
        else:
            consec = 0
            convergence_start = None
    return None


def convergence_summary(agent_type, bar_type, cfg_template, seeds, metric="attendance",
                         window=200, rel_tol=0.02, sustain=3):
    """Runs `seeds` independent simulations for one pairing at
    cfg_template.rounds, applies convergence_round to each seed's
    round-by-round `metric` series, and summarizes across seeds: how
    many converged within the horizon at all, and the
    median/min/max round at which they did (for those that did).
    Import is local to avoid a circular import with simulation.py.
    """
    from simulation import run_simulation

    conv_rounds = []
    n_seeds = len(seeds)
    for seed in seeds:
        cfg = copy.deepcopy(cfg_template)
        cfg.seed = seed
        df, agents, bar = run_simulation(agent_type, bar_type, cfg=cfg, seed=seed)
        cr = convergence_round(df[metric].values, window=window, rel_tol=rel_tol, sustain=sustain)
        if cr is not None:
            conv_rounds.append(cr)

    n_converged = len(conv_rounds)
    return {
        "agent_type": agent_type, "bar_type": bar_type, "metric": metric,
        "rounds_simulated": cfg_template.rounds, "n_seeds": n_seeds,
        "n_converged": n_converged,
        "fraction_converged": n_converged / n_seeds if n_seeds else np.nan,
        "median_convergence_round": float(np.median(conv_rounds)) if conv_rounds else np.nan,
        "min_convergence_round": float(np.min(conv_rounds)) if conv_rounds else np.nan,
        "max_convergence_round": float(np.max(conv_rounds)) if conv_rounds else np.nan,
    }
    """Average empirical regret (see empirical_regret) across every agent
    in a no-regret population for a single run. Reporting a single agent
    from a single run (as the original figure/analysis did) is not
    representative of the population; this averages across all n_agents
    in one run. Returns (mean_regret, std_regret_across_agents)."""
    regrets = [empirical_regret(a, realized_K, realized_price, cfg)[0] for a in agents]
    return float(np.mean(regrets)), float(np.std(regrets))


def multi_seed_population_regret(agent_type, bar_type, cfg_template, seeds, window=50):
    """Rolling population-average regret trajectory, averaged again across
    multiple independent seeds. Requires running fresh simulations (one
    per seed), since regret depends on each seed's own realized play.
    Returns a list of (round, mean_regret_across_seeds, std_regret_across_seeds),
    where the mean/std at each round are taken over (n_agents x n_seeds)
    individual-agent regret values.
    Import is local to avoid a circular import with simulation.py.
    """
    from simulation import run_simulation

    per_seed_trajectories = []
    checkpoints_ref = None
    for seed in seeds:
        cfg = copy.deepcopy(cfg_template)
        cfg.seed = seed
        df, agents, bar = run_simulation(agent_type, bar_type, cfg=cfg, seed=seed)
        n = len(df)
        checkpoints = list(range(window, n + 1, max(1, window // 5)))
        if checkpoints_ref is None:
            checkpoints_ref = checkpoints
        traj = []
        for end in checkpoints:
            agent_regrets = [
                empirical_regret(a, df["attendance"].values[:end],
                                  df["price"].values[:end], cfg)[0]
                for a in agents
            ]
            traj.append(agent_regrets)  # list of n_agents regrets at this checkpoint
        per_seed_trajectories.append(traj)  # shape: [checkpoints][n_agents]

    out = []
    for c_idx, end in enumerate(checkpoints_ref):
        pooled = []
        for seed_traj in per_seed_trajectories:
            pooled.extend(seed_traj[c_idx])
        out.append((end, float(np.mean(pooled)), float(np.std(pooled))))
    return out
