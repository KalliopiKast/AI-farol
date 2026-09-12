"""R3.1 public-feedback baseline at full scale (100 seeds, 1000 rounds),
for heuristic and Bayesian agents against all four bar types, with welfare
computed the same way as the main grid (metrics.welfare_series)."""
import sys, os
import pandas as pd
from model import GameConfig
from simulation import run_simulation
from metrics import welfare_series

agent_type, bar_type, n_seeds = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for seed in range(n_seeds):
    cfg = GameConfig(rounds=1000, seed=seed)
    df, agents, bar = run_simulation(agent_type, bar_type, cfg=cfg, seed=seed,
                                      agent_kwargs={"use_public_feedback": True})
    _, _, welfare = welfare_series(df["attendance"], df["price"], cfg)
    rows.append({"agent_type": agent_type, "bar_type": bar_type, "seed": seed,
                 "attendance_mean": df["attendance"].mean(),
                 "price_mean": df["price"].mean(),
                 "profit_mean": df["profit"].mean(),
                 "welfare_mean": welfare.mean()})
out = pd.DataFrame(rows)
path = "results/r3_1_public_feedback_full.csv"
write_header = not os.path.exists(path)
out.to_csv(path, mode="a", header=write_header, index=False)
print(f"DONE: {agent_type}|{bar_type}  K={out.attendance_mean.mean():.3f}  Pi={out.profit_mean.mean():.3f}  W={out.welfare_mean.mean():.3f}")
