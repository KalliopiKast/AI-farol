import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
N_AGENTS = 10
THRESHOLD = 6
ROUNDS = 100
PRICE = 10

attendance_history = []
satisfaction_history = []
income_history = []

memory_lengths = [random.randint(2, 5) for _ in range(N_AGENTS)]

def predict_attendance(history, memory):
    if len(history) < memory:
        return random.randint(0, N_AGENTS)
    return int(np.mean(history[-memory:]))

for t in range(ROUNDS):
    decisions = []

    for i in range(N_AGENTS):
        pred = predict_attendance(attendance_history, memory_lengths[i])
        decisions.append(1 if pred < THRESHOLD else 0)

    attendance = sum(decisions)
    attendance_history.append(attendance)

    satisfaction = np.mean([1 if (d == 1 and attendance <= THRESHOLD) else 0 for d in decisions])
    satisfaction_history.append(satisfaction)

    income_history.append(attendance * PRICE)

# Export CSV
df = pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
})
df.to_csv("baseline_results.csv", index=False)

# Plots
fig, axs = plt.subplots(3, 1)

axs[0].plot(attendance_history)
#axs[0].axhline(y=THRESHOLD)
axs[0].set_title("Attendance")

axs[1].plot(satisfaction_history)
axs[1].set_title("Satisfaction")

axs[2].plot(income_history)
axs[2].set_title("Income")

plt.show()