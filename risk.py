"""
Tail-risk metrics: Value-at-Risk and Conditional VaR (Expected Shortfall).

We define everything in *loss* space. Given a P&L sample, loss = -pnl.

  VaR_alpha   = the alpha-quantile of the loss distribution.
                "We are alpha-confident the loss won't exceed this."
  CVaR_alpha  = E[loss | loss >= VaR_alpha], the average of the worst (1-alpha)
                tail. CVaR is coherent (sub-additive) where VaR is not, which is
                why it's the metric Basel and most risk desks actually prefer.

Both are reported as positive numbers representing a loss.
"""

from __future__ import annotations

import numpy as np


def var_cvar(pnl, alpha=0.95):
    """Historical (non-parametric) VaR and CVaR from a P&L sample."""
    losses = -np.asarray(pnl, dtype=float)
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if tail.size else var
    return float(var), float(cvar)


def risk_summary(pnl, alphas=(0.95, 0.99)):
    """Dict of summary risk stats for a P&L sample."""
    out = {
        "mean": float(np.mean(pnl)),
        "std": float(np.std(pnl, ddof=1)),
        "min": float(np.min(pnl)),
        "max": float(np.max(pnl)),
    }
    for a in alphas:
        v, c = var_cvar(pnl, a)
        out[f"VaR_{int(a*100)}"] = v
        out[f"CVaR_{int(a*100)}"] = c
    return out
