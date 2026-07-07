"""
agents.py
Agent-side learning mechanisms for the AI-Farol model (Section 3.2).

All agents share the interface:
    decide(t, price) -> int (0 or 1)
    observe(t, price, K_t, attended) -> None

PARTIAL OBSERVABILITY is implemented literally as in the paper: an agent
only records a data point (K_t, price_t) in its PRIVATE history for
periods in which it attended (a_i^t = 1). Agents that never attend never
learn anything about attendance levels from this channel. This creates
the selection bias flagged explicitly in Section 3.2 ("agent i has
observations only for periods in which they attended, potentially
introducing selection bias").

A separate PUBLIC-DISCLOSURE channel is available for learning algorithms
that need full feedback to function (no-regret, Q-learning): it represents
a bar that publishes aggregate attendance after the fact (e.g. a
next-day announcement), independent of the real-time partial observation
during the visit itself. This is a modeling choice, made explicit here so
it can be toggled off (`use_public_feedback=False`) to isolate the effect
of pure selection-biased, private-history learning -- e.g. to reproduce
the classical partial-observability regime for the HeuristicMemoryAgent
and BayesianAgent.
"""

import numpy as np
from model import satisfaction


class BaseAgent:
    def __init__(self, agent_id, cfg, use_public_feedback=True, rng=None):
        self.id = agent_id
        self.cfg = cfg
        self.use_public_feedback = use_public_feedback
        self.rng = rng or np.random.default_rng()
        self.private_history = []   # (t, K_t, price_t), only when attended
        self.public_history = []    # (t, K_t, price_t), if disclosure is on
        self.utility_history = []   # realized u_i^t each round

    def _history(self):
        return self.public_history if self.use_public_feedback else self.private_history

    def observe(self, t, price, K_t, attended):
        if attended:
            self.private_history.append((t, K_t, price))
        if self.use_public_feedback:
            self.public_history.append((t, K_t, price))
        u = attended * (satisfaction(K_t, self.cfg) - price)
        self.utility_history.append(float(u))

    def decide(self, t, price):
        raise NotImplementedError


class HeuristicMemoryAgent(BaseAgent):
    """Classical bounded-rationality agent (baseline El Farol heuristic).
    Predicts next attendance as the mean of the last `memory` observed
    values (subject to selection bias if use_public_feedback=False), then
    attends only if the prediction is below the target threshold.
    """
    def __init__(self, agent_id, cfg, memory=None, **kwargs):
        super().__init__(agent_id, cfg, **kwargs)
        self.memory = memory or int(self.rng.integers(2, 6))

    def decide(self, t, price):
        hist = self._history()
        if len(hist) < self.memory:
            pred = self.rng.integers(0, self.cfg.n_agents + 1)
        else:
            pred = np.mean([h[1] for h in hist[-self.memory:]])
        return 1 if pred < self.cfg.k_target else 0


class BayesianAgent(BaseAgent):
    """Forms a Beta-Bernoulli belief over the population attendance rate
    from (potentially selection-biased) observations -- the Eq. (11)
    belief-formation channel -- then decides probabilistically via a
    sigmoid over predicted congestion and price.
    """
    def __init__(self, agent_id, cfg, sensitivity=None, **kwargs):
        super().__init__(agent_id, cfg, **kwargs)
        self.alpha = 1.0  # Beta prior pseudo-counts (attended-like mass)
        self.beta = 1.0
        self.sensitivity = sensitivity if sensitivity is not None else self.rng.uniform(0.3, 0.7)

    def _update_belief(self, K_t):
        rate = K_t / self.cfg.n_agents
        self.alpha += rate
        self.beta += (1 - rate)

    def decide(self, t, price):
        hist = self._history()
        if hist:
            self._update_belief(hist[-1][1])
        p_hat = self.alpha / (self.alpha + self.beta)
        pred_K = p_hat * self.cfg.n_agents
        utility = self.cfg.k_target - pred_K - self.sensitivity * price
        prob = 1 / (1 + np.exp(-utility))
        return 1 if self.rng.random() < prob else 0


class NoRegretAgent(BaseAgent):
    """Multiplicative-Weights (Hedge) no-regret learner over a small set of
    canned strategies, following Hart & Mas-Colell (2000) as cited in the
    paper. If a full population of such agents learns simultaneously, the
    time-averaged joint distribution of play should approach a coarse
    correlated equilibrium (empirically checked via metrics.empirical_regret).

    NOTE: strategy evaluation uses the REALIZED K_t of the round, which
    requires the public-disclosure channel (use_public_feedback=True,
    the default). This is standard in no-regret learning: the learner
    needs to know what every strategy *would have* earned, not just what
    it actually earned.
    """
    def __init__(self, agent_id, cfg, eta=0.15, **kwargs):
        kwargs.setdefault("use_public_feedback", True)
        super().__init__(agent_id, cfg, **kwargs)
        self.eta = eta
        self.strategies = self._make_strategies()
        self.weights = np.ones(len(self.strategies))

    def _make_strategies(self):
        target = self.cfg.k_target
        return [
            lambda h: 1,                                                       # always attend
            lambda h: 0,                                                       # never attend
            lambda h: 1 if (not h or h[-1][1] < target) else 0,                 # attend if last K below target
            lambda h: 1 if (h and h[-1][1] >= target) else 0,                   # contrarian: attend if crowded last time
            lambda h: 1 if (len(h) >= 2 and np.mean([x[1] for x in h[-2:]]) < target) else 0,  # 2-period average
        ]

    def decide(self, t, price):
        hist = self._history()
        probs = self.weights / self.weights.sum()
        choice = self.rng.choice(len(self.strategies), p=probs)
        self._last_choice = choice
        return int(self.strategies[choice](hist))

    def observe(self, t, price, K_t, attended):
        # snapshot history BEFORE this round's outcome is appended, so
        # counterfactual strategies are evaluated on the same information
        # the agent actually had when it chose an action this round.
        hist_before = list(self._history())
        super().observe(t, price, K_t, attended)
        if not self.use_public_feedback:
            return
        rewards = np.array([
            strat(hist_before) * (satisfaction(K_t, self.cfg) - price)
            for strat in self.strategies
        ])
        span = rewards.max() - rewards.min()
        r_norm = (rewards - rewards.min()) / (span + 1e-9)
        self.weights *= np.exp(self.eta * r_norm)
        self.weights /= self.weights.sum()


class QLearningAgent(BaseAgent):
    """Individual tabular Q-learning agent over a discretized
    (last-observed attendance, price bucket) state space, attend/not-attend
    actions, reward = realized utility u_i^t (Section 3.2).
    """
    def __init__(self, agent_id, cfg, price_buckets, alpha=0.1, gamma=0.9,
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05, **kwargs):
        kwargs.setdefault("use_public_feedback", True)
        super().__init__(agent_id, cfg, **kwargs)
        self.price_buckets = np.asarray(price_buckets)
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        n_states = cfg.n_agents + 1
        self.Q = np.zeros((n_states, len(self.price_buckets), 2))
        self._last_state = None
        self._last_action = None
        self._last_price_idx = None

    def _price_index(self, price):
        return int(np.argmin(np.abs(self.price_buckets - price)))

    def _state(self):
        hist = self._history()
        return int(hist[-1][1]) if hist else self.cfg.k_target

    def decide(self, t, price):
        s = self._state()
        p_idx = self._price_index(price)
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(0, 2))
        else:
            a = int(np.argmax(self.Q[s, p_idx]))
        self._last_state, self._last_price_idx, self._last_action = s, p_idx, a
        return a

    def observe(self, t, price, K_t, attended):
        super().observe(t, price, K_t, attended)
        reward = attended * (satisfaction(K_t, self.cfg) - price)
        s, p_idx, a = self._last_state, self._last_price_idx, self._last_action
        s_next = int(K_t)
        best_next = np.max(self.Q[s_next])
        self.Q[s, p_idx, a] += self.alpha * (reward + self.gamma * best_next - self.Q[s, p_idx, a])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


AGENT_REGISTRY = {
    "heuristic": HeuristicMemoryAgent,
    "bayesian": BayesianAgent,
    "no_regret": NoRegretAgent,
    "q_learning": QLearningAgent,
}
