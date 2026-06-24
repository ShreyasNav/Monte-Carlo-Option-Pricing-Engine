# Monte Carlo Option Pricer with Variance Reduction

A European option pricing engine that values calls and puts two independent ways —
the closed-form **Black–Scholes–Merton** formula and a **Monte Carlo** simulator —
cross-validates them, and quantifies how much **variance-reduction** techniques cut
the simulation cost. It also computes the full set of Greeks and the tail risk
(VaR / CVaR) of an option position.

The point of the project is not "Monte Carlo can price an option" (it can; everyone
knows that). The point is the engineering question underneath it: *a naive simulation
is needlessly expensive, so how much compute can you buy back with the standard
variance-reduction tricks, and which ones actually work for this payoff?*

## Headline results

Test contract: `S = K = 100`, `r = 5%`, `σ = 20%`, `T = 1y`, European call.

| quantity | value |
|---|---|
| Black–Scholes analytical price | **10.4506** (matches Hull's textbook) |
| Monte Carlo (control variate, 200k paths) | 10.4377 ± 0.0125 — agrees within its **95% confidence interval** |
| **Control-variate variance reduction** | **≈ 6.9×** → same accuracy with ~7× fewer paths |
| Antithetic-variate variance reduction | **≈ 2.0×** |
| Greeks at the money | Δ 0.637, Γ 0.019, vega 37.5, θ −6.41, ρ 53.2 |
| 95% VaR / CVaR of a short call (real-world μ=10%) | 39.5 / 52.9 per contract |

### Why control variates beat antithetic variates here

Both techniques reduce variance, and the project *measures* each rather than asserting it.

Antithetic sampling pairs every draw `Z` with `−Z` and averages the two payoffs. It
helps whenever the payoff is **monotonic** in `Z`, because then `f(Z)` and `f(−Z)` are
negatively correlated and the averaging cancels part of the noise. A call payoff is
monotonic in `Z`, so antithetic genuinely helps — about **2×** here. (The gain is bounded
because the payoff is also zero over the entire out-of-the-money half, which weakens the
cancellation.)

The control variate does better (~7×) because it subtracts the discounted terminal stock
`e^{−rT}Sₜ`, whose true mean is known exactly (`S₀`) and which is **highly correlated**
with the call payoff. The stronger the correlation between the control and the payoff, the
more variance is removed — and the terminal stock is a much stronger control than the
symmetry antithetic exploits.

## What's inside

```
src/
  black_scholes.py   analytical price + all five Greeks (vectorized)
  monte_carlo.py     plain / antithetic / control-variate estimators, CIs, convergence
  risk.py            historical VaR and CVaR (expected shortfall)
scripts/
  run_analysis.py    runs everything, prints results, writes plots/ and results/
tests/
  test_pricing.py    textbook value, put-call parity, MC-in-CI, delta vs finite-difference
plots/               four figures (see below)
```

## Figures

- `01_convergence.png` — MC estimate and 95% band funnelling onto the analytical price.
- `02_variance_reduction.png` — standard error vs path count for all three estimators (log-log).
- `03_greeks.png` — delta and gamma across spot.
- `04_tail_risk.png` — short-call P&L distribution with VaR and CVaR marked.

## Design choices a reviewer might ask about

- **Exact terminal sampling, not Euler stepping.** For a European payoff only `Sₜ`
  matters, and GBM has an exact lognormal solution, so there is no discretisation bias
  to introduce — every path is drawn from the true terminal law in one step.
- **Control variate beta is estimated from the same sample.** A textbook simplification;
  it introduces a small bias that vanishes as `n → ∞` and is negligible at the path
  counts used. Noted here rather than hidden.
- **Risk is measured under the physical measure, pricing under the risk-neutral one.**
  The option premium is the risk-neutral (Q) fair value you actually receive, but VaR/CVaR
  ask "how much can I lose in the real world," so the underlying is simulated under an
  assumed real-world drift (P), not the risk-free rate. The mean short-call P&L is mildly
  negative because, with no volatility risk premium modelled, a higher real drift makes the
  short call finish in the money more often.
- **Risk is reported in loss space, and CVaR alongside VaR.** CVaR is coherent
  (sub-additive) where VaR is not, which is why risk desks and Basel prefer it; both are
  shown so the difference is visible in the tail.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_analysis.py     # results + plots
python tests/test_pricing.py       # correctness checks
```
