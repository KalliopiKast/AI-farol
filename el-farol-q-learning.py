import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PARAMETERS
N_AGENTS = 10
TARGET = 6
#ROUNDS = 100
ROUNDS = 100
#prices = [0.5, 1.0, 2.0, 3.0]
prices = [3, 5, 7, 10]
Q = np.zeros((N_AGENTS + 1, len(prices)))  # states: 0..10 attendance

alpha = 0.1
gamma = 0.9
epsilon = 0.3
epsilon_decay = 0.995
epsilon_min = 0.05

# DATA STORAGE
attendance_history = []
satisfaction_history = []
income_history = []

# Agent heterogeneity
agent_sensitivity = np.random.uniform(0.3, 0.7, N_AGENTS)

# --- Probabilistic agent decision ---
def agent_decision(predicted_attendance, price, i):
    utility = TARGET - predicted_attendance - agent_sensitivity[i] * price
    prob = 1 / (1 + np.exp(-utility))  # sigmoid
    return 1 if random.random() < prob else 0

# INITIAL STATE
state = TARGET  # start from balanced expectation

# --- Simulation ---
for t in range(ROUNDS):

    # --- Action selection (epsilon-greedy) ---
    if random.random() < epsilon:
        action = random.randint(0, len(prices) - 1)
    else:
        action = np.argmax(Q[state])

    price = prices[action]

    # --- Agents decide ---
    decisions = [agent_decision(state, price, i) for i in range(N_AGENTS)]
    attendance = sum(decisions)

    # --- Reward (better balanced) ---
    crowd_penalty = -abs(attendance - TARGET)
    revenue_reward = 0.2 * attendance * price
    reward = crowd_penalty + revenue_reward

    # --- Q-learning update ---
    next_state = attendance
    Q[state, action] += alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state, action]
    )

    state = next_state

    # --- Decay exploration ---
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # --- Store data ---
    attendance_history.append(attendance)

    # --- Graded satisfaction ---
    satisfaction = np.mean([
        max(0, 1 - abs(attendance - TARGET) / TARGET) if d == 1 else 0
        for d in decisions
    ])
    satisfaction_history.append(satisfaction)

    income_history.append(attendance * price)

# --- Export CSV ---
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("improved_q_learning_results.csv", index=False)

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