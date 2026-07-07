import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# PARAMETERS
N_AGENTS = 10
TARGET = 6
#ROUNDS = 100
ROUNDS = 100
#BASE_PRICE = 1.0
BASE_PRICE = 10

# DATA STORAGE
attendance_history = []
satisfaction_history = []
income_history = []

X_data, y_data = [], []
model = LinearRegression()

# Agent heterogeneity (each agent reacts differently to price)
agent_sensitivity = np.random.uniform(0.3, 0.7, N_AGENTS)

# --- Agent decision (probabilistic) ---
def agent_decision(predicted, price, i):
    utility = TARGET - predicted - agent_sensitivity[i] * price
    
    # Sigmoid to convert utility -> probability
    prob = 1 / (1 + np.exp(-utility))
    
    return 1 if random.random() < prob else 0

# --- Simulation ---
price = BASE_PRICE

for t in range(ROUNDS):

    # --- Prediction ---
    if len(X_data) > 5:
        model.fit(X_data, y_data)
        pred = model.predict(np.array(attendance_history[-3:]).reshape(1, -1))[0]
    else:
        pred = random.randint(0, N_AGENTS)

    # --- Bar pricing strategy (smooth adjustment) ---
    error = TARGET - pred
    price += 0.05 * (-error)   # raise price if too many expected
    price = max(0.1, min(5, price))  # clamp price

    # --- Agents decide ---
    decisions = [agent_decision(pred, price, i) for i in range(N_AGENTS)]
    attendance = sum(decisions)
    attendance_history.append(attendance)

    # --- Train regression ---
    if len(attendance_history) >= 3:
        X_data.append(attendance_history[-3:])
        y_data.append(attendance)

    # --- Satisfaction (graded, not binary) ---
    satisfaction = np.mean([
        max(0, 1 - abs(attendance - TARGET) / TARGET) if d == 1 else 0
        for d in decisions
    ])
    satisfaction_history.append(satisfaction)

    # --- Income ---
    income_history.append(attendance * price)

# --- Export CSV ---
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("improved_linear_regression_results.csv", index=False)

# --- Plots ---
fig, axs = plt.subplots(3, 1, figsize=(15, 4))

axs[0].plot(attendance_history)
#axs[0].axhline(y=TARGET, linestyle='--')
axs[0].set_title("Attendance")

axs[1].plot(satisfaction_history)
axs[1].set_title("Satisfaction")

axs[2].plot(income_history)
axs[2].set_title("Income")

plt.tight_layout()
plt.show()