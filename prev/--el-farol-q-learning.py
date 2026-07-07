import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N_AGENTS = 10
TARGET = 6
ROUNDS = 100

prices = [0.5, 1.0, 2.0, 3.0]
Q = np.zeros((11, len(prices)))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

attendance_history = []
satisfaction_history = []
income_history = []

def agent_decision(attendance, price):
    return 1 if (TARGET - attendance - 0.5 * price) > 0 else 0

state = 5

for t in range(ROUNDS):

    action = random.randint(0, len(prices)-1) if random.random() < epsilon else np.argmax(Q[state])
    price = prices[action]

    decisions = [agent_decision(state, price) for _ in range(N_AGENTS)]
    attendance = sum(decisions)

    reward = -abs(attendance - TARGET) + 0.1 * attendance

    next_state = attendance
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    state = next_state

    attendance_history.append(attendance)

    satisfaction = np.mean([1 if (d == 1 and attendance <= TARGET) else 0 for d in decisions])
    satisfaction_history.append(satisfaction)

    income_history.append(attendance * price)

# Export CSV
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("q_learning_results.csv", index=False)

# Plots
fig, axs = plt.subplots(1, 3)

axs[0].plot(attendance_history)
axs[0].axhline(y=TARGET)
axs[0].set_title("Attendance")

axs[1].plot(satisfaction_history)
axs[1].set_title("Satisfaction")

axs[2].plot(income_history)
axs[2].set_title("Income")

plt.show()