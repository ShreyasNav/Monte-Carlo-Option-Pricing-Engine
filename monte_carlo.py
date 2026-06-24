"""
Monte Carlo pricing of European options with variance reduction.

Three estimators are implemented so we can quantify *how much* each trick buys:

  - plain          : naive risk-neutral terminal sampling
  - antithetic     : pair each Z with -Z; kills the odd part of the payoff
  - control_variate : use the discounted terminal stock S_T as a control,
                      whose true mean E[S_T] = S0 * exp((r - q) T) is known.

The headline number an interviewer cares about is the *variance-reduction
factor* (ratio of estimator variances at fixed sample size), so we report it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    ci95: tuple[float, float]
    n_paths: int
    method: str

    def __repr__(self):
        lo, hi = self.ci95
        return (f"MCResult({self.method}: {self.price:.5f} "
                f"+/- {self.std_error:.5f}, 95% CI [{lo:.5f}, {hi:.5f}], "
                f"n={self.n_paths})")


def _terminal_prices(S0, r, q, sigma, T, Z):
    """Exact one-step GBM terminal price under the risk-neutral measure."""
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def _payoff(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def price_mc(S0, K, r, q, sigma, T, kind="call", n_paths=100_000,
             method="control_variate", seed=None):
    """Monte Carlo price with the chosen variance-reduction method."""
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if method == "antithetic":
        # Draw m pairs (Z, -Z). The estimator averages the m *pair averages*,
        # which ARE iid; the variance of the estimator must be computed from
        # those pair averages, NOT from the 2m individual payoffs (those are
        # negatively correlated, and that correlation is the whole benefit).
        m = n_paths // 2
        Z = rng.standard_normal(m)
        ST_pos = _terminal_prices(S0, r, q, sigma, T, Z)
        ST_neg = _terminal_prices(S0, r, q, sigma, T, -Z)
        pay_pos = disc * _payoff(ST_pos, K, kind)
        pay_neg = disc * _payoff(ST_neg, K, kind)
        samples = 0.5 * (pay_pos + pay_neg)   # one iid sample per pair
        n_used = 2 * m
    else:
        Z = rng.standard_normal(n_paths)
        ST = _terminal_prices(S0, r, q, sigma, T, Z)
        discounted_payoff = disc * _payoff(ST, K, kind)
        if method == "plain":
            samples = discounted_payoff
        elif method == "control_variate":
            # Control = discounted terminal stock, a martingale whose mean is
            # known exactly: E[e^{-rT} S_T] = S0 e^{-qT}. Highly correlated with
            # the payoff, so subtracting it strips out most of the variance.
            control = disc * ST
            mean_control = S0 * np.exp(-q * T)
            cov = np.cov(discounted_payoff, control)
            beta = cov[0, 1] / cov[1, 1]
            samples = discounted_payoff - beta * (control - mean_control)
        else:
            raise ValueError(f"unknown method {method!r}")
        n_used = n_paths

    est = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(len(samples))
    return MCResult(price=est, std_error=se,
                    ci95=(est - 1.96 * se, est + 1.96 * se),
                    n_paths=n_used, method=method)


def variance_reduction_factor(S0, K, r, q, sigma, T, kind="call",
                              n_paths=100_000, seed=0):
    """Ratio Var(plain) / Var(method) at fixed n. >1 means the trick helps."""
    base = price_mc(S0, K, r, q, sigma, T, kind, n_paths, "plain", seed)
    out = {}
    for m in ("antithetic", "control_variate"):
        res = price_mc(S0, K, r, q, sigma, T, kind, n_paths, m, seed)
        # Variance scales as SE^2 * n; n matches, so compare SE^2 directly.
        out[m] = (base.std_error / res.std_error) ** 2
    return out


def convergence_curve(S0, K, r, q, sigma, T, kind="call",
                      method="plain", path_grid=None, seed=0):
    """Estimate and standard error as a function of sample size n."""
    if path_grid is None:
        path_grid = np.unique(np.logspace(2, 6, 25).astype(int))
    ns, prices, ses = [], [], []
    for n in path_grid:
        res = price_mc(S0, K, r, q, sigma, T, kind, int(n), method, seed)
        ns.append(res.n_paths)
        prices.append(res.price)
        ses.append(res.std_error)
    return np.array(ns), np.array(prices), np.array(ses)
