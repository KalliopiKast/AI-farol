# AI-Farol: Simulation Code

Companion code for *"AI-Farol: Co-Evolutionary Dynamics in a Multi-Agent
Two-Sided Learning Framework"*.

This repository implements the two-sided dynamic game described in the
paper (Section 2): a population of boundedly rational agents deciding
whether to attend a bar under partial observability, and a bar that sets
prices strategically using AI-based demand forecasting and policy
learning.

## What's here

| File | Paper section | Contents |
|---|---|---|
| `model.py` | Sec. 2.1, Eq. 1-3 | Satisfaction `U(K)`, bar cost `C_B(K)`, utility, profit, welfare, constraints |
| `agents.py` | Sec. 3.2 | Agent-side learning: heuristic memory (classical baseline), Bayesian belief formation, no-regret (multiplicative weights), Q-learning |
| `bar.py` | Sec. 3.1 | Bar-side learning: passive (classical), linear-regression forecasting, Gaussian-Process Bayesian forecasting, Q-learning pricing with constraint penalties |
| `simulation.py` | Sec. 2.2, Eq. 4-7 | The exact 7-step game loop |
| `metrics.py` | Sec. 4 | Welfare decomposition (Eq. 13-14) and empirical no-regret / CCE convergence diagnostics |
| `run_experiments.py` | — | Runs the full agent × bar grid, saves CSVs and comparison plots to `results/` |

## Design notes / modeling choices made explicit

- **Partial observability** is implemented literally: an agent only
  records a private data point for periods it actually attended
  (`private_history`). This creates the selection bias the paper flags
  in Section 3.2.
- **Public disclosure channel.** No-regret and Q-learning agents need to
  evaluate counterfactual strategies/rewards against the *realized*
  attendance, which requires the bar to publish aggregate attendance
  after the fact (e.g. a next-day report) — a separate, explicit
  assumption (`use_public_feedback=True`, default for these two agent
  types) from the real-time partial observation during the visit itself.
  Set `use_public_feedback=False` on the heuristic/Bayesian agents to
  study the pure selection-bias regime.
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
python run_experiments.py
```

This runs every `{agent_type} × {bar_type}` combination (4 agent types ×
up to 4 bar types) for 300 rounds each, saving:
- one CSV per combination in `results/`
- comparison plots (`comparison_attendance.png`, `comparison_price.png`,
  `comparison_profit.png`, `comparison_social_welfare.png`)
- a rolling no-regret convergence plot for each `no_regret__*` bar
  combination (`regret_*.png`), the empirical check for convergence
  toward a coarse correlated equilibrium (Hart & Mas-Colell, 2000)

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
  regret/CCE check, not a theoretical guarantee.
- The clustering-based demand forecaster from earlier drafts was dropped
  in favor of the Gaussian Process bar, which more directly instantiates
  the "Bayesian... risk-aware, exploration/exploitation" claim in
  Section 3.1 and gives an actual uncertainty estimate.

## Citation

If you use this code, please cite the paper (see main repository /
publication for the current BibTeX entry).
