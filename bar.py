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
    `lag` attendance values AND the (contemporaneous) price -- the
    Eq. (8) demand-forecasting bar, K_hat_t = f(p_t, K_{t-1}, ..., K_{t-M}).

    Feature vector is [K_{t-lag}, ..., K_{t-1}, p_t] (price is always the
    LAST column; see _price_coefficient). Training pairs come from the
    bar's own price/attendance history, so the regressor only has
    something to fit once its own pricing has produced some price
    variation -- which the early-round fallback below (identical to the
    lag-only rule used before enough data has accumulated) already
    supplies, without requiring a separate forced-exploration term.

    Once fitted, price is set by solving the linear model for the price
    that would put predicted attendance at k_target, damped by `step` so
    the bar moves toward that price gradually rather than jumping to it.

    PERFORMANCE NOTE: refitting (and re-converting the ever-growing
    self._X/self._y lists to numpy arrays) on EVERY round -- as both this
    bar and the original, pre-review-fix version did -- costs O(t) per
    round and O(T^2) over a run. At 1000 rounds this was invisible
    (~1s); at 10,000 rounds it dominated the entire grid (~30-38s per
    run, worse than the windowed GP bar). Refitting is gated behind
    `refit_every`, mirroring GPBar; that alone only reduces the constant
    (still fitting on an unboundedly growing dataset each time it does
    refit, so total cost was still growing with T, ~7-10s at 10,000
    rounds). The fit is now ALSO bounded to the most recent `window`
    samples, exactly as in GPBar, which is what actually caps per-refit
    cost independent of total round count.

    As with GPBar, this windowing is NOT behaviorally neutral: a
    window-bounded OLS fit is a genuinely different (locally
    re-estimated, forgetting old regimes) model from one fit on the
    complete history, particularly at long horizons where the two
    diverge most. Results from this version should not be presented as
    numerically identical to an unbounded-history fit.

    ABLATION: pass use_price_feature=False to reproduce the price-blind
    regression (lags only, no price regressor) for the R3.3 before/after
    comparison in Section 5.9 -- everything else (refit cadence, window,
    early-round fallback) stays identical to the corrected default, so
    the comparison isolates the effect of adding price as a feature
    rather than conflating it with the other changes made this revision.
    """
    def __init__(self, cfg, lag=3, step=0.4, price_min=None, price_max=None,
                 refit_every=5, window=200, use_price_feature=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.lag = lag
        self.step = step
        self.price = cfg.p0
        self.price_min = price_min if price_min is not None else cfg.p0 - cfg.d_max
        self.price_max = price_max if price_max is not None else cfg.p0
        self.refit_every = refit_every
        self.window = window
        self.use_price_feature = use_price_feature
        self.model = LinearRegression()
        self._X, self._y = [], []
        self._fitted = False

    def set_price(self, t):
        if len(self.K_history) < self.lag or len(self._X) < 5:
            # not enough data to fit a price-aware model yet: fall back to
            # the original lag-only error-correction rule. This is also
            # what generates the initial price variation the regression
            # below needs in order to identify a price coefficient at all.
            pred = self.K_history[-1] if self.K_history else self.cfg.k_target
            error = self.cfg.k_target - pred
            self.price += self.step * (-error) * 0.1
            self.price = float(np.clip(self.price, self.price_min, self.price_max))
            return self.price

        if t % self.refit_every == 0 or not self._fitted:
            X_win = np.array(self._X[-self.window:])
            y_win = np.array(self._y[-self.window:])
            self.model.fit(X_win, y_win)
            self._fitted = True

        lag_feats = self.K_history[-self.lag:]
        intercept = self.model.intercept_
        coefs = self.model.coef_

        if self.use_price_feature:
            # [.., coef_lagM, coef_price] -- price is the last column
            lag_contrib = intercept + sum(c * k for c, k in zip(coefs[:-1], lag_feats))
            price_coef = coefs[-1]
        else:
            # price-blind ablation: no price column at all, so the fitted
            # prediction does not depend on price -- there is no price
            # coefficient to solve for, only the lag-only fallback below.
            lag_contrib = intercept + sum(c * k for c, k in zip(coefs, lag_feats))
            price_coef = 0.0

        if abs(price_coef) > 1e-6:
            target_price = (self.cfg.k_target - lag_contrib) / price_coef
            self.price += self.step * (target_price - self.price)
        else:
            # price effect not (yet) identifiable from the fitted model
            # (or, in the price-blind ablation, not modeled at all):
            # fall back to the lag-only error-correction rule.
            error = self.cfg.k_target - lag_contrib
            self.price += self.step * (-error) * 0.1

        self.price = float(np.clip(self.price, self.price_min, self.price_max))
        return self.price

    def observe(self, t, K_t, price_t, profit_t):
        if len(self.K_history) >= self.lag:
            feats = list(self.K_history[-self.lag:])
            if self.use_price_feature:
                feats = feats + [price_t]
            self._X.append(feats)
            self._y.append(K_t)
        super().observe(t, K_t, price_t, profit_t)


class GPBar(BaseBar):
    """Bayesian demand forecasting via Gaussian Process regression on
    (price, K_{t-1}) -> K_t, with an upper-confidence-bound rule for
    choosing among candidate prices at the current K_{t-1}. Realizes the
    "Bayesian approach ... risk-aware decisions that balance exploration
    and exploitation" claim of Section 3.1.

    Feature vector is [price_t, K_{t-1}] -- K_{t-1} is the attendance
    observed the round BEFORE the one being priced, captured in
    observe() before that round's own K_t is appended to K_history.
    This mirrors RegressionBar's inclusion of a lagged-attendance
    feature: without it, RegressionBar sees (price + lags) while GPBar
    saw price alone, an information asymmetry that confounded the
    Section 5.3 forecasting-bar comparison the RQ2 research question
    rests on (regression could out- or under-perform GP for reasons
    having nothing to do with which forecasting METHOD is better). The
    two bars now condition on the same information.

    PERFORMANCE NOTE: the GP is refit on only the most recent `window`
    observations rather than the entire accumulated history. A GP fit's
    cost scales roughly cubically in the number of training points (kernel
    hyperparameter optimization over an NxN covariance matrix), so
    refitting on an unbounded, ever-growing history made long runs
    (~150s for a single 1000-round run) impractical for a full seed x
    pairing x horizon grid; windowing brings this down to ~2s.

    IMPORTANT CAVEAT: windowing is NOT behaviorally neutral. A spot check
    against the unbounded version (same seed, 300 rounds) showed the
    bar's own price trajectory can differ substantially between the two
    (up to the full candidate-price range), since the GP is genuinely
    fit on different data. For agent types whose decisions do not depend
    on price at all (HeuristicMemoryAgent, NoRegretAgent -- see their
    strategy definitions), this has no effect on attendance outcomes.
    For BayesianAgent and QLearningAgent, which DO condition on price,
    windowing is expected to shift outcomes to some degree and should be
    treated as a genuine (if reasonable -- an unbounded training set is
    itself not obviously more realistic for an actual venue) change to
    the bar's behavior, not a pure speed optimization. Any GP-bar results
    generated with this version should not be presented as numerically
    identical to results from the original unbounded implementation.

    MODELING NOTE: the RBF kernel uses a single scalar length_scale
    shared across both feature dimensions rather than an
    automatic-relevance-determination (per-dimension) length scale.
    Since price and K_{t-1} both range over roughly [0, 10] in this
    configuration, this is not a severe simplification here, but would
    need revisiting if the two features' natural scales diverged.
    """
    def __init__(self, cfg, candidate_prices=None, kappa=1.0, refit_every=10,
                 window=150, **kwargs):
        super().__init__(cfg, **kwargs)
        if not _HAS_GP:
            raise ImportError("scikit-learn's GaussianProcessRegressor is required for GPBar")
        self.candidates = np.asarray(
            candidate_prices if candidate_prices is not None
            else np.linspace(cfg.p0 - cfg.d_max, cfg.p0, 6)
        )
        self.kappa = kappa
        self.refit_every = refit_every
        self.window = window
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._X, self._y = [], []
        self._fitted = False

    def set_price(self, t):
        if len(self._X) < 5:
            return float(self.rng.choice(self.candidates))
        if t % self.refit_every == 0 or not self._fitted:
            X_win = np.array(self._X[-self.window:])
            y_win = np.array(self._y[-self.window:])
            self.gp.fit(X_win, y_win)
            self._fitted = True
        k_lag = self.K_history[-1] if self.K_history else self.cfg.k_target
        cand_features = np.column_stack(
            [self.candidates, np.full_like(self.candidates, k_lag)])
        mu, sigma = self.gp.predict(cand_features, return_std=True)
        expected_K = np.clip(mu, 0, self.cfg.n_agents)
        expected_profit = self.candidates * expected_K - (self.cfg.c_fixed + self.cfg.c_var * expected_K)
        score = expected_profit + self.kappa * sigma
        return float(self.candidates[int(np.argmax(score))])

    def observe(self, t, K_t, price_t, profit_t):
        k_lag = self.K_history[-1] if self.K_history else self.cfg.k_target
        self._X.append([price_t, k_lag])
        self._y.append(K_t)
        super().observe(t, K_t, price_t, profit_t)


class QLearningBar(BaseBar):
    """Markov-decision-process bar (Eq. 9): tabular Q-learning over
    (previous attendance) state and a discrete set of price actions.
    Reward = profit, with penalty terms for constraint violations
    (Eq. 3) built directly into the reward signal, as described in
    Section 3.1 ("operational constraints are incorporated as penalty
    terms in the reward signal").

    Action set: 5 evenly-spaced discount levels spanning [0, d_max]
    (previously 3 fixed prices). A coarser 3-point action set could
    understate how well policy learning can do relative to the passive
    forecasting bars -- and this pairing (no_regret/q_learning-bar,
    q_learning/q_learning-bar) is the mechanism behind the paper's most
    load-bearing empirical claim (only the Q-learning bar recovers
    attendance from sophisticated agents, Section 5.3), so giving it a
    genuinely fair action space matters more here than for a peripheral
    result.

    State discretization deliberately left as the raw last-observed K_t
    integer (not binned into five, and with no separate profit
    dimension): profit is a deterministic function of (state, action) in
    this model (profit = price*K - cost(K)), so a profit-bucket state
    dimension would be redundant with information the (state, action)
    pair already encodes, not a genuine missing capability.
    """
    def __init__(self, cfg, prices=None, alpha=0.1, gamma=0.9, epsilon=0.3,
                 epsilon_decay=0.995, epsilon_min=0.05, penalty=10.0, **kwargs):
        super().__init__(cfg, **kwargs)
        self.prices = prices or list(np.linspace(cfg.p0 - cfg.d_max, cfg.p0, 5))
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
