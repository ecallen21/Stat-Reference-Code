# Concentration Inequalities (Reference §46.14)

Boucheron, Lugosi & Massart (2013). Deterministic bounds on the
probability that a random variable deviates from its mean.

## Classical inequalities (bounded r.v.s in `[a, b]`)

| Bound | Formula |
|---|---|
| **Markov** | `P(X ≥ t) ≤ E[X] / t` |
| **Chebyshev** | `P(|X − μ| ≥ t) ≤ Var(X) / t²` |
| **Hoeffding** | `P(|S_n − ES_n| ≥ t) ≤ 2 exp(−2 t² / (n(b−a)²))` |
| **Bernstein** | `P(|S_n − ES_n| ≥ t) ≤ 2 exp(−t² / (2·n·σ² + 2Mt/3))` |
| **McDiarmid** | Bounded-differences generalisation of Hoeffding |

## Files

- `python/concentration_inequalities.py` — analytic bounds + Monte-
  Carlo empirical probabilities. Demo (n=200 Bernoulli(0.3)): shows
  Hoeffding, Bernstein, Chebyshev bounds vs empirical tail across
  t ∈ {5, 10, 20, 30, 40}. Bernstein tightest at low variance;
  Chebyshev tightest at small t; all valid upper bounds.
- `r/concentration_inequalities.R` — from-scratch; no CRAN package.

## When to use

- **Finite-sample guarantees** — proving generalisation, PAC, MAB
  regret, robustness.
- **Confidence intervals without distributional assumptions** —
  bounded r.v.s + Hoeffding.
- **Design of adaptive experiments / bandits** — UCB / KL-UCB regret
  bounds are direct applications.

## When NOT to use

- **Practical CI reporting for well-known distributions** — asymptotic
  CLT / bootstrap is tighter.
- **Unbounded / heavy-tailed variables** — Hoeffding fails; use
  Bernstein with sub-exponential tails or truncation.

## Assumptions & caveats

- **Independence** — most classical bounds need iid; martingale
  extensions (Azuma-Hoeffding, McDiarmid) relax this.
- **Boundedness** — Hoeffding needs `X ∈ [a, b]`; for
  sub-Gaussian / sub-exponential tails use Bernstein / Bennett.
- **Tightness** — bounds are worst-case; empirical tail typically
  smaller.
- **Two-sided vs one-sided** — halve the exponent for one-sided.

## Related in this repo

- `rademacher-complexity`, `vc-dimension`, `efron-stein-inequality`
  — used with concentration inequalities in PAC proofs.
- `multi-armed-bandits` — UCB uses Hoeffding directly.
- `sequential-analysis`, `always-valid-inference` — anytime-valid
  cousins built from concentration.

## Run

```
python techniques/concentration-inequalities/python/concentration_inequalities.py
Rscript techniques/concentration-inequalities/r/concentration_inequalities.R
```

**Refs:** Boucheron, S., Lugosi, G. & Massart, P. *Concentration Inequalities: A Nonasymptotic Theory of Independence*, OUP, 2013; Hoeffding, W. "Probability inequalities for sums of bounded random variables." *JASA*, 58(301): 13-30, 1963; Bernstein, S.N. *The Theory of Probabilities*, 1946.

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
