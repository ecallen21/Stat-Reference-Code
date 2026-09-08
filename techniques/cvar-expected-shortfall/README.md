# CVaR / Expected Shortfall (Reference §47.42)

Rockafellar & Uryasev (2000). Coherent risk measure at confidence α:

    VaR_α(X) = inf { x : P(X ≤ x) ≥ α }
    CVaR_α(X) = E[X | X ≥ VaR_α(X)]

Unlike VaR, CVaR is **subadditive** (satisfies Artzner et al 1999
coherence axioms). Rockafellar-Uryasev representation

    CVaR_α = min_c  c + (1 − α)⁻¹ E[(X − c)₊]

is CONVEX in c (and in decisions when X depends on them) — enables
convex portfolio LPs / QPs.

## Files

- `python/cvar_expected_shortfall.py` — from-scratch empirical
  VaR, tail-mean ES, and RU-form CVaR via `scipy.optimize.
  minimize_scalar`. Demo (t₃ losses, n=20 000):
  RU-CVaR matches tail-mean ES within Monte-Carlo error across
  α ∈ {0.90, 0.95, 0.99}.
- `r/cvar_expected_shortfall.R` — `PerformanceAnalytics::ES`,
  `quantreg::rq` for QR-based CVaR (R); `empyrical`, `cvxpy`,
  from-scratch (Python).

## When to use

- **Portfolio optimisation** — Rockafellar-Uryasev LP.
- **Tail-risk hedging** — reinsurance, credit, ops risk.
- **Basel III / Solvency II** — regulatory ES has replaced VaR.
- **DRO** — CVaR is the dual of a Wasserstein DRO ball.

## When NOT to use

- **Very small samples** — heavy-tail ES estimates are noisy;
  use POT-EVT or L-moments for tail extrapolation.
- **Non-loss objectives** — apply to X = −return, or flip signs.
- **Non-convex decision variables** — the RU trick only preserves
  convexity for linear cost mappings.

## Assumptions & caveats

- **Loss convention** — assumes larger X = worse; flip signs for
  returns.
- **Empirical ES** biased downward at small n; use bootstrap CIs
  or POT / EVT extrapolation.
- **Coherent axioms** — monotonicity, subadditivity, positive
  homogeneity, translation invariance.
- **Different α conventions** — some texts use `1 − α` (loss quantile
  vs safety level); watch the sign.

## Related in this repo

- `extreme-value-theory` — POT / GEV for the deep tail.
- `distributionally-robust-optimization` — CVaR ≡ Wasserstein-DRO
  under a size-α ball.
- `bayesian-quantile-regression`, `quantile-regression` —
  distributional risk quantities.
- `copulas` — joint tail dependence for portfolio ES.

## Run

```
python techniques/cvar-expected-shortfall/python/cvar_expected_shortfall.py
Rscript techniques/cvar-expected-shortfall/r/cvar_expected_shortfall.R
```

**Refs:** Rockafellar, R.T. & Uryasev, S. "Optimization of conditional value-at-risk." *Journal of Risk* 2(3): 21-41, 2000; Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. "Coherent measures of risk." *Math Finance* 9(3): 203-228, 1999.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
