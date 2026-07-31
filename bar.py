"""
bar.py
Bar-side AI mechanisms for the AI-Farol model (Section 3.1).

All bars share the interface:
    set_price(t) -> float
    observe(t, K_t, price_t, profit_t) -> None

The bar chooses a discount d_t in [0, d_max], with p_t = p0 - d_t (Eq. 1).
Operational constraints (Eq. 3) are enforced via penalty terms in the
reward signal for the learning-based bars, and every bar tracks
`distress_periods` -- the count of rounds violating the profitability
constraint Pi_min -- so simulations can report near-bankruptcy episodes.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    _HAS_GP = True
except ImportError:
    _HAS_GP = False


class BaseBar:
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.K_history = []
        self.price_history = []
        self.profit_history = []
        self.distress_periods = 0  # rounds violating Pi_min

    def set_price(self, t):
        raise NotImplementedError

    def observe(self, t, K_t, price_t, profit_t):
        self.K_history.append(K_t)
        self.price_history.append(price_t)
        self.profit_history.append(profit_t)
        if profit_t < self.cfg.pi_min:
            self.distress_periods += 1


class PassiveBar(BaseBar):
    """Classical passive venue: fixed price, no adaptation whatsoever."""
    def set_price(self, t):
        return self.cfg.p0


class RegressionBar(BaseBar):
    """Forecasts next-period attendance via linear regression on the last
    `lag` attendance values, then adjusts price smoothly toward the target
    occupancy -- the Eq. (8) demand-forecasting bar.
    """
    def __init__(self, cfg, lag=3, step=0.4, price_min=None, price_max=None, **kwargs):
        super().__init__(cfg, **kwargs)
        self.lag = lag
        self.step = step
        self.price = cfg.p0
        self.price_min = price_min if price_min is not None else cfg.p0 - cfg.d_max
        self.price_max = price_max if price_max is not None else cfg.p0
        self.model = LinearRegression()
        self._X, self._y = [], []

    def _predict(self):
        if len(self.K_history) < self.lag or len(self._X) < 5:
            return self.K_history[-1] if self.K_history else self.cfg.k_target
        self.model.fit(np.array(self._X), np.array(self._y))
        feat = np.array(self.K_history[-self.lag:]).reshape(1, -1)
        return float(self.model.predict(feat)[0])

    def set_price(self, t):
        pred = self._predict()
        error = self.cfg.k_target - pred
        self.price += self.step * (-error) * 0.1
        self.price = float(np.clip(self.price, self.price_min, self.price_max))
        return self.price

    def observe(self, t, K_t, price_t, profit_t):
        if len(self.K_history) >= self.lag:
            self._X.append(self.K_history[-self.lag:])
            self._y.append(K_t)
        super().observe(t, K_t, price_t, profit_t)


class GPBar(BaseBar):
    """Bayesian demand forecasting via Gaussian Process regression on
    (price -> attendance), with an upper-confidence-bound rule for
    choosing among candidate prices. Realizes the "Bayesian approach ...
    risk-aware decisions that balance exploration and exploitation" claim
    of Section 3.1.
    """
    def __init__(self, cfg, candidate_prices=None, kappa=1.0, refit_every=5, **kwargs):
        super().__init__(cfg, **kwargs)
        if not _HAS_GP:
            raise ImportError("scikit-learn's GaussianProcessRegressor is required for GPBar")
        self.candidates = np.asarray(
            candidate_prices if candidate_prices is not None
            else np.linspace(cfg.p0 - cfg.d_max, cfg.p0, 6)
        )
        self.kappa = kappa
        self.refit_every = refit_every
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._X, self._y = [], []
        self._fitted = False

    def set_price(self, t):
        if len(self._X) < 5:
            return float(self.rng.choice(self.candidates))
        if t % self.refit_every == 0 or not self._fitted:
            self.gp.fit(np.array(self._X), np.array(self._y))
            self._fitted = True
        mu, sigma = self.gp.predict(self.candidates.reshape(-1, 1), return_std=True)
        expected_K = np.clip(mu, 0, self.cfg.n_agents)
        expected_profit = self.candidates * expected_K - (self.cfg.c_fixed + self.cfg.c_var * expected_K)
        score = expected_profit + self.kappa * sigma
        return float(self.candidates[int(np.argmax(score))])

    def observe(self, t, K_t, price_t, profit_t):
        self._X.append([price_t])
        self._y.append(K_t)
        super().observe(t, K_t, price_t, profit_t)


class QLearningBar(BaseBar):
    """Markov-decision-process bar (Eq. 9): tabular Q-learning over
    (previous attendance) state and a discrete set of price actions.
    Reward = profit, with penalty terms for constraint violations
    (Eq. 3) built directly into the reward signal, as described in
    Section 3.1 ("operational constraints are incorporated as penalty
    terms in the reward signal").
    """
    def __init__(self, cfg, prices=None, alpha=0.1, gamma=0.9, epsilon=0.3,
                 epsilon_decay=0.995, epsilon_min=0.05, penalty=10.0, **kwargs):
        super().__init__(cfg, **kwargs)
        self.prices = prices or [cfg.p0 - cfg.d_max, cfg.p0 - cfg.d_max / 2, cfg.p0]
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.penalty = penalty
        n_states = cfg.n_agents + 1
        self.Q = np.zeros((n_states, len(self.prices)))
        self.state = cfg.k_target
        self._last_action = None

    def set_price(self, t):
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(0, len(self.prices)))
        else:
            a = int(np.argmax(self.Q[self.state]))
        self._last_action = a
        return self.prices[a]

    def observe(self, t, K_t, price_t, profit_t):
        reward = profit_t
        if not (self.cfg.k_min <= K_t <= self.cfg.k_max):
            reward -= self.penalty
        if profit_t < self.cfg.pi_min:
            reward -= self.penalty
        s, a = self.state, self._last_action
        s_next = int(K_t)
        self.Q[s, a] += self.alpha * (reward + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])
        self.state = s_next
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        super().observe(t, K_t, price_t, profit_t)


BAR_REGISTRY = {
    "passive": PassiveBar,
    "regression": RegressionBar,
    "gp": GPBar,
    "q_learning": QLearningBar,
}
