"""
Correctness tests. Run with:  python -m pytest tests/ -q  (or just python tests/test_pricing.py)

These pin the engine to facts that must hold regardless of implementation:
  - the textbook Black-Scholes value,
  - put-call parity,
  - Monte Carlo agreeing with the analytical price inside its own 95% CI,
  - the analytical delta matching a finite-difference bump.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import black_scholes as bs
from src import monte_carlo as mc

PARAMS = dict(S=100.0, K=100.0, r=0.05, q=0.0, sigma=0.20, T=1.0)


def test_black_scholes_textbook_value():
    c = bs.price(100, 100, 0.05, 0.0, 0.20, 1.0, "call")
    assert abs(c - 10.4506) < 1e-3, c


def test_put_call_parity():
    p = PARAMS
    c = bs.price(p["S"], p["K"], p["r"], p["q"], p["sigma"], p["T"], "call")
    pu = bs.price(p["S"], p["K"], p["r"], p["q"], p["sigma"], p["T"], "put")
    parity = c - pu - (p["S"] * np.exp(-p["q"] * p["T"])
                       - p["K"] * np.exp(-p["r"] * p["T"]))
    assert abs(parity) < 1e-10, parity


def test_mc_matches_bs_within_ci():
    p = PARAMS
    truth = bs.price(p["S"], p["K"], p["r"], p["q"], p["sigma"], p["T"], "call")
    res = mc.price_mc(p["S"], p["K"], p["r"], p["q"], p["sigma"], p["T"],
                      "call", n_paths=400_000, method="control_variate", seed=0)
    assert res.ci95[0] <= truth <= res.ci95[1], (res, truth)


def test_delta_matches_finite_difference():
    p = PARAMS
    h = 1e-4
    up = bs.price(p["S"] + h, p["K"], p["r"], p["q"], p["sigma"], p["T"], "call")
    dn = bs.price(p["S"] - h, p["K"], p["r"], p["q"], p["sigma"], p["T"], "call")
    fd = (up - dn) / (2 * h)
    ana = bs.delta(p["S"], p["K"], p["r"], p["q"], p["sigma"], p["T"], "call")
    assert abs(fd - ana) < 1e-6, (fd, ana)


def test_variance_reduction_factors():
    p = PARAMS
    vrf = mc.variance_reduction_factor(p["S"], p["K"], p["r"], p["q"],
                                       p["sigma"], p["T"], "call",
                                       n_paths=200_000, seed=0)
    # Antithetic helps because the call payoff is monotonic in Z; control
    # variate helps more because the terminal stock is a strong control.
    assert vrf["antithetic"] > 1.3, vrf
    assert vrf["control_variate"] > 3.0, vrf
    assert vrf["control_variate"] > vrf["antithetic"], vrf


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
    print("\nAll tests passed.")
