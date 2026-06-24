"""
Analytical Black-Scholes-Merton pricing and Greeks.

Everything here is closed-form. It serves two purposes in this project:
  1. A ground-truth benchmark to validate the Monte Carlo engine against.
  2. The delta source for the hedging simulator (we re-price and re-delta at
     every rebalancing date).

Convention: continuous dividend yield q, risk-free rate r, vol sigma, all
annualized. Time T in years.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, q, sigma, T):
    S, K, sigma, T = map(np.asarray, (S, K, sigma, T))
    # Guard against T -> 0 to avoid division blow-ups at expiry.
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def price(S, K, r, q, sigma, T, kind="call"):
    """Black-Scholes price of a European option.

    `kind` is "call" or "put". Vectorized over any broadcastable inputs.
    """
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif kind == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def delta(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dS. This is the hedge ratio the simulator rebalances to."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, r, q, sigma, T):
    """d2Price/dS2. Same for calls and puts. Drives the hedging error."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrtT)


def vega(S, K, r, q, sigma, T):
    """dPrice/dsigma, per unit (1.00) of vol. Divide by 100 for per-1%-vol."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    return S * np.exp(-q * T) * norm.pdf(d1) * sqrtT


def theta(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dt (per year). Negative for long options, the cost of carry."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    term1 = -S * disc_q * norm.pdf(d1) * sigma / (2 * sqrtT)
    if kind == "call":
        return term1 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    return term1 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)


def rho(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dr, per unit of rate."""
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    if kind == "call":
        return K * T * disc_r * norm.cdf(d2)
    return -K * T * disc_r * norm.cdf(-d2)


def all_greeks(S, K, r, q, sigma, T, kind="call"):
    """Convenience: return every Greek in one dict."""
    return {
        "price": price(S, K, r, q, sigma, T, kind),
        "delta": delta(S, K, r, q, sigma, T, kind),
        "gamma": gamma(S, K, r, q, sigma, T),
        "vega": vega(S, K, r, q, sigma, T),
        "theta": theta(S, K, r, q, sigma, T, kind),
        "rho": rho(S, K, r, q, sigma, T, kind),
    }
