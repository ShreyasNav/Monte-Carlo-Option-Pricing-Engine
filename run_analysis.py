"""
Run the full analysis: validate the Monte Carlo engine against Black-Scholes,
quantify the variance-reduction speedup, compute the Greeks, and measure the
tail risk of an option position. Saves figures to plots/ and a results table
to results/.

Run:  python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import black_scholes as bs        # noqa: E402
from src import monte_carlo as mc          # noqa: E402
from src import risk                       # noqa: E402

PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# A single, clearly stated contract everything below uses.
S0, K, r, q, SIGMA, T = 100.0, 100.0, 0.05, 0.0, 0.20, 1.0
KIND = "call"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"


def section(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def validate_and_variance_reduction():
    section("1. VALIDATION + VARIANCE REDUCTION")
    bs_price = float(bs.price(S0, K, r, q, SIGMA, T, KIND))
    print(f"Black-Scholes analytical price : {bs_price:.6f}")

    summary = {"bs_price": bs_price, "estimators": {}}
    for m in ("plain", "antithetic", "control_variate"):
        res = mc.price_mc(S0, K, r, q, SIGMA, T, KIND, n_paths=200_000,
                          method=m, seed=7)
        in_ci = res.ci95[0] <= bs_price <= res.ci95[1]
        print(f"  {m:16s} {res.price:.5f}  SE={res.std_error:.5f}  "
              f"BS in 95% CI: {in_ci}")
        summary["estimators"][m] = {"price": res.price, "se": res.std_error}

    vrf = mc.variance_reduction_factor(S0, K, r, q, SIGMA, T, KIND,
                                       n_paths=200_000, seed=7)
    summary["variance_reduction_factor"] = vrf
    print("\nVariance-reduction factor vs plain MC (higher = better):")
    for m, f in vrf.items():
        print(f"  {m:16s} {f:5.2f}x  ->  same accuracy with ~{f:.0f}x "
              f"fewer paths / less compute")
    return summary


def plot_convergence():
    section("2. CONVERGENCE")
    bs_price = float(bs.price(S0, K, r, q, SIGMA, T, KIND))
    ns, prices, ses = mc.convergence_curve(S0, K, r, q, SIGMA, T, KIND,
                                           method="plain", seed=3)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.fill_between(ns, prices - 1.96 * ses, prices + 1.96 * ses,
                    color=BLUE, alpha=0.18, label="95% confidence band")
    ax.plot(ns, prices, color=BLUE, lw=1.6, label="Monte Carlo estimate")
    ax.axhline(bs_price, color=RED, lw=1.4, ls="--",
               label=f"Black-Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of simulated paths")
    ax.set_ylabel("call price")
    ax.set_title("Monte Carlo converges to the analytical price")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "01_convergence.png")
    plt.close(fig)
    print("saved plots/01_convergence.png")


def plot_variance_reduction():
    section("3. VARIANCE-REDUCTION CURVES")
    grid = np.unique(np.logspace(2.5, 6, 18).astype(int))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {"plain": BLUE, "antithetic": ORANGE, "control_variate": GREEN}
    for m, c in colors.items():
        ns, _, ses = mc.convergence_curve(S0, K, r, q, SIGMA, T, KIND,
                                          method=m, path_grid=grid, seed=3)
        ax.plot(ns, ses, "o-", ms=3, color=c, lw=1.4,
                label=m.replace("_", " "))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of simulated paths")
    ax.set_ylabel("standard error of price estimate")
    ax.set_title("Control variates cut the standard error at every budget")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "02_variance_reduction.png")
    plt.close(fig)
    print("saved plots/02_variance_reduction.png")


def plot_greeks():
    section("4. GREEKS")
    spots = np.linspace(60, 140, 200)
    delta = bs.delta(spots, K, r, q, SIGMA, T, KIND)
    gamma = bs.gamma(spots, K, r, q, SIGMA, T)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot(spots, delta, color=BLUE, lw=1.8)
    a1.axvline(K, color="gray", ls=":", lw=1)
    a1.set_title("Delta"); a1.set_xlabel("spot S"); a1.set_ylabel("dV/dS")
    a2.plot(spots, gamma, color=GREEN, lw=1.8)
    a2.axvline(K, color="gray", ls=":", lw=1)
    a2.set_title("Gamma (concentrated near the strike)")
    a2.set_xlabel("spot S"); a2.set_ylabel("d2V/dS2")
    fig.suptitle("Option sensitivities for the ATM 1Y call", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_greeks.png", bbox_inches="tight")
    plt.close(fig)

    g = bs.all_greeks(S0, K, r, q, SIGMA, T, KIND)
    print("Greeks at S=K=100:")
    for k, v in g.items():
        print(f"  {k:6s} {v: .5f}")
    print("saved plots/03_greeks.png")
    return {k: float(v) for k, v in g.items()}


def plot_risk():
    section("5. TAIL RISK OF A SHORT-CALL POSITION")
    # P&L at maturity of selling one ATM call: collect the (risk-neutral) fair
    # premium, then owe the realized payoff. VaR/CVaR are REAL-WORLD risk
    # measures, so the underlying is simulated under the physical measure with
    # an assumed real-world drift MU -- not the risk-free rate. This is the
    # distinction that matters: you price under Q, you measure risk under P.
    MU = 0.10  # assumed real-world equity drift
    rng = np.random.default_rng(11)
    n = 200_000
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((MU - q - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * Z)
    premium = float(bs.price(S0, K, r, q, SIGMA, T, KIND))  # fair Q-price received
    payoff = np.maximum(ST - K, 0.0)
    pnl = premium * np.exp(r * T) - payoff   # seller's P&L at T, under P
    print(f"(real-world drift assumed: mu = {MU:.0%}; vol = {SIGMA:.0%})")

    stats = risk.risk_summary(pnl)
    for k, v in stats.items():
        print(f"  {k:9s} {v: .4f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hist(pnl, bins=120, color=BLUE, alpha=0.7, density=True)
    ax.axvline(-stats["VaR_95"], color=ORANGE, lw=1.6,
               label=f"95% VaR = {stats['VaR_95']:.2f}")
    ax.axvline(-stats["CVaR_95"], color=RED, lw=1.6,
               label=f"95% CVaR = {stats['CVaR_95']:.2f}")
    ax.set_xlabel("P&L at maturity (per contract)")
    ax.set_ylabel("density")
    ax.set_title("Short-call P&L distribution and tail risk")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_tail_risk.png")
    plt.close(fig)
    print("saved plots/04_tail_risk.png")
    return stats


def main():
    print("Contract:  S0=K=100,  r=5%,  sigma=20%,  T=1y,  European call")
    results = {
        "contract": {"S0": S0, "K": K, "r": r, "q": q,
                     "sigma": SIGMA, "T": T, "kind": KIND},
        "pricing": validate_and_variance_reduction(),
    }
    plot_convergence()
    plot_variance_reduction()
    results["greeks"] = plot_greeks()
    results["risk"] = plot_risk()

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/summary.json")


if __name__ == "__main__":
    main()
