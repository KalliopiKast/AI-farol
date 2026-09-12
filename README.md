# AI-Farol: Simulation Code

Companion code for *"AI-Farol: Co-Evolutionary Dynamics in a Multi-Agent
Two-Sided Learning Framework"*.

This repository implements the two-sided dynamic game described in the
paper (Section 2): a population of boundedly rational agents deciding
whether to attend a bar under partial observability, and a bar that sets
prices strategically using AI-based demand forecasting and policy
learning.

## Post-review revisions (read this first)

Three implementation-vs-model discrepancies were identified during peer
review and are fixed in this version. If you are comparing against
results from the originally submitted manuscript, note that **all
numerical results need to be regenerated** — none of the three fixes
below are expected to leave the original numbers unchanged.

1. **Partial observability was not actually exercised by the experiment
   grid (`agents.py`, `simulation.py`).** `BaseAgent`'s default was
   `use_public_feedback=True`, and the experiment runner never overrode
   it, so the heuristic and Bayesian agent types — the ones meant to
   demonstrate the selection-biased, partially-observable regime of
   Section 2.2 / Eq. (6), (11) — ran under full public disclosure of
   `K_t` in the original grid. Fixed by changing the `BaseAgent` default
   to `use_public_feedback=False` (no-regret and Q-learning agents
   explicitly opt back into `True`, since their counterfactual-strategy
   evaluation structurally needs the realized outcome). Additionally, the
   private-history channel previously stored the *true* `K_t` on nights
   an agent attended — a weaker, aggregate-only notion of partial
   observability than the literal subset `S_i^t` of Eq. (6)/(11). This is
   now implemented literally: `simulation.py` samples a fixed-size subset
   of the *other* agents each round (`GameConfig.obs_subset_size`,
   default 3) and gives an attending agent only a rescaled estimate built
   from that subset, never the true aggregate.
2. **The regression bar never used price as a regressor (`bar.py`).**
   Eq. (8) is `K_hat_t = f(p_t, K_{t-1}, ..., K_{t-M})`; the
   implementation regressed only on lagged attendance. `RegressionBar`
   now includes contemporaneous price as the last feature column, and
   sets price by solving the fitted linear model for the price that
   targets `k_target` attendance (damped by `step`), rather than the
   previous lag-only error-correction rule. Early rounds (before 5
   training points exist) still use the original lag-only rule, which
   is also what seeds the price variation the regression needs to
   identify a price coefficient at all.
3. **The no-regret empirical-regret calculation reused the realized
   `K_t` for every candidate strategy (`metrics.py`).** A candidate
   strategy that would have played a different action than the agent's
   realized action also changes total attendance, and hence the
   congestion-sensitive payoff, in that round. `empirical_regret` now
   recomputes the exact one-round counterfactual attendance
   `K_t' = K_t - a_i^t + a'` (valid since actions within a round are
   simultaneous, so every OTHER agent's contribution to `K_t` is
   unaffected by this agent's counterfactual) before scoring each
   candidate strategy. `agents.py` now tracks each agent's own realized
   action history (`action_history`) to support this. Two further
   additions: `population_regret` averages regret across every agent
   in a run rather than reporting one; `multi_seed_population_regret`
   averages that again across seeds. Both replace the single-run,
   single-agent regret figure from the original submission (see
   `run_experiments.run_regret_analysis`).

A fourth, non-reviewer-flagged issue was found and fixed during this
pass: `GPBar` refit on its entire accumulated history on every
`refit_every`-th round, which made a single 1000-round run take ~150s
(a full seed × pairing × horizon grid at that cost was not going to
finish). It now refits on a bounded sliding window
(`GPBar(window=150)` by default), which brings a 1000-round run down to
~2s. **This is not behaviorally neutral** — see the `GPBar` docstring in
`bar.py` for a spot-checked description of what changes and for which
agent types.

## What's here

| File | Paper section | Contents |
|---|---|---|
| `model.py` | Sec. 2.1, Eq. 1-3 | Satisfaction `U(K)`, bar cost `C_B(K)`, utility, profit, welfare, constraints, `obs_subset_size` config |
| `agents.py` | Sec. 3.2 | Agent-side learning: heuristic memory (classical baseline), Bayesian belief formation, no-regret (multiplicative weights), Q-learning. Literal subset-based partial observability (see above). |
| `bar.py` | Sec. 3.1 | Bar-side learning: passive (classical), price-aware linear-regression forecasting, windowed Gaussian-Process Bayesian forecasting, Q-learning pricing with constraint penalties |
| `simulation.py` | Sec. 2.2, Eq. 4-7 | The exact 7-step game loop, including per-round subset generation for step (6) |
| `metrics.py` | Sec. 4 | Welfare decomposition (Eq. 13-14); corrected empirical no-regret / CCE convergence diagnostics, including population- and multi-seed-averaged regret |
| `run_experiments.py` | — | Runs the full agent × bar grid, saves raw per-seed and aggregated-summary CSVs, convergence plots, and the corrected regret analysis to `results/` |
| `analyze_results.py` | — | Post-hoc statistics on `results/raw_all_rounds.csv`: 95% CIs (normal-theory and bootstrap), Welch t-tests on the paper's headline comparisons, and a CV-vs-horizon convergence-diagnostic slope per pairing |

## Design notes / modeling choices made explicit

- **Partial observability** is implemented via a literal subset
  mechanism: an attending agent observes only a size-`obs_subset_size`
  sample of the other agents and infers a rescaled attendance estimate
  from it, never the true `K_t`. Agents that never attend learn nothing
  from this channel at all. This creates the selection bias flagged in
  Section 3.2, now compounded with genuine within-period partial
  information. Set `cfg.obs_subset_size = None` to fall back to the
  weaker aggregate-only regime (true `K_t` recorded on nights attended).
- **Public disclosure channel.** No-regret and Q-learning agents need to
  evaluate counterfactual strategies/rewards against the *realized*
  attendance, which requires the bar to publish aggregate attendance
  after the fact (e.g. a next-day report) — a separate, explicit
  assumption (`use_public_feedback=True`, the default for these two
  agent types only) from the real-time partial observation during the
  visit itself.
- **Two-sided learning.** Earlier drafts of this codebase only had the
  bar learning (agents used a fixed heuristic). This version adds
  genuine agent-side learning (`no_regret`, `q_learning`) so the
  co-evolutionary dynamics of Section 3.3 can actually be demonstrated
  empirically, not just asserted.
- **Constraints (Eq. 3)** are tracked via `bar.distress_periods` (rounds
  violating `Pi_min`) for every bar type, and folded into the reward
  signal directly for `QLearningBar`.

## Running it

```bash
pip install -r requirements.txt
python run_experiments.py        # seed count: AIFAROL_N_SEEDS env var, default 100
python analyze_results.py        # requires results/raw_all_rounds.csv from the step above
```

`run_experiments.py` runs every `{agent_type} × {bar_type}` combination
(4 agent types × up to 4 bar types) at three round horizons
(10 / 100 / 1000), `AIFAROL_N_SEEDS` times each (default 100; override
with e.g. `AIFAROL_N_SEEDS=50 python run_experiments.py`), and saves to
`results/`:
- `raw_all_rounds.csv` — one row per (agent_type, bar_type, rounds,
  seed): the actual per-seed values needed for real statistical work
- `summary_all_rounds.csv` and `summary_{rounds}rounds.csv` — aggregated
  mean/std per pairing, for quick inspection
- `convergence_{agent}_{bar}_{metric}.png` and
  `combined_convergence_{metric}.png` — how each metric's mean/std
  evolves across the three horizons, for four illustrative pairings
- `regret_corrected_no_regret.png` / `.csv` — population- and
  multi-seed-averaged empirical regret for `no_regret` agents against
  every bar type, using the corrected counterfactual (see above)

`analyze_results.py` then reads `raw_all_rounds.csv` and writes:
- `ci_summary.csv` — normal-theory and percentile-bootstrap 95% CIs per
  (agent_type, bar_type, rounds, metric) cell; cells with coefficient of
  variation > 1 are flagged, and the bootstrap CI (not the normal-theory
  one) should be treated as primary for those
- `welch_tests.csv` — Welch's t-test (unequal variances) on the specific
  headline comparisons the paper's Section 5 discussion relies on (edit
  `HEADLINE_COMPARISONS` in the script to add more)
- `cv_slopes.csv` — OLS slope of coefficient-of-variation against
  log10(rounds) per pairing/metric: negative means dispersion shrinks as
  the horizon grows (consistent with settling), positive means it grows
  (consistent with the persistent-adaptation / limit-cycle regime of
  Sec. 3.3 / 5.5)

To run a single combination interactively:

```python
from model import GameConfig
from simulation import run_simulation

cfg = GameConfig(rounds=300, seed=0)
df, agents, bar = run_simulation(agent_type="no_regret", bar_type="q_learning", cfg=cfg)
print(df.tail())
print("distress periods:", bar.distress_periods)
```

## Known simplifications (see paper Sec. 5, "Limitations")

- Homogeneous agents beyond a single sensitivity parameter; no explicit
  agent types/segments.
- Single bar, single pricing instrument (no capacity/information-
  disclosure control, though the config supports adding these).
- No formal convergence proof; `metrics.py` provides an *empirical*
  regret/CCE check (now with the counterfactual correction described
  above), not a theoretical guarantee.
- The clustering-based demand forecaster from earlier drafts was dropped
  in favor of the Gaussian Process bar, which more directly instantiates
  the "Bayesian... risk-aware, exploration/exploitation" claim in
  Section 3.1 and gives an actual uncertainty estimate.
- The literal subset-observability mechanism (`obs_subset_size`) samples
  subsets uniformly at random each round; it does not model persistent
  social ties (the same agents tending to observe the same others round
  after round), which a more realistic extension might.

## Citation

If you use this code, please cite the paper (see main repository /
publication for the current BibTeX entry).
