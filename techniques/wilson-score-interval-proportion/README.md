# Wilson / Agresti-Coull / Jeffreys / Clopper-Pearson CIs (Reference §4.18)

Wilson (1927); Agresti & Coull (1998); Brown, Cai & DasGupta (2001).
Better confidence intervals for a binomial proportion than the
classical Wald interval, especially near `p = 0` or `p = 1`.

## Formulae

| Method | Interval |
|---|---|
| **Wald** | `p̂ ± z · √(p̂(1−p̂)/n)` |
| **Wilson score** | `((2n p̂ + z²) ± z √(z² + 4n p̂(1−p̂))) / (2(n + z²))` |
| **Agresti-Coull** | Wald with `p̃ = (x + z²/2)/(n + z²)` |
| **Jeffreys** | Beta(x + ½, n − x + ½) quantiles |
| **Clopper-Pearson** | Exact Beta CI |

## Files

- `python/wilson_score_interval_proportion.py` — all five methods
  from scratch. Demo (x, n):
  - (0, 10): Wald (0, 0) fails, Wilson (0, 0.278), Jeffreys (0, 0.217)
  - (3, 10): Wald (0.02, 0.58) undercovers, Wilson (0.11, 0.60)
  - (30, 100): all similar (0.21, 0.40)
  - (98, 100): Wald (0.953, 1.0) is misleading, Wilson (0.930, 0.994)
- `r/wilson_score_interval_proportion.R` — `binom::binom.confint`,
  `PropCIs`, `stats::prop.test` (R);
  `statsmodels.stats.proportion.proportion_confint`,
  `scipy.stats.binomtest` (Python).

## When to use

- **Any proportion CI reporting** — Wilson is the modern default.
- **Small n or extreme p** — Wald is unreliable; use Wilson,
  Agresti-Coull, Jeffreys.
- **Exact coverage required** — Clopper-Pearson (conservative but
  guaranteed ≥ 1 − α).
- **Regulatory / clinical** — Jeffreys is Bayesian-plausible with
  reference prior.

## When NOT to use

- **Continuous outcomes** — CI for a mean uses t / bootstrap, not
  proportion methods.
- **Multi-category** — use multinomial CIs (Goodman, Sison-Glaz).
- **Clustered / weighted data** — extend via design-based SE
  (survey package).

## Assumptions & caveats

- **Independence** — assumed for all methods; violated by clustering.
- **Wald degeneracy at 0/n** — gives (0, 0) or (1, 1); avoid.
- **Coverage vs interval length** — Wilson/AC target average
  coverage 1-α; Clopper-Pearson guarantees ≥ 1-α (wider).
- **Continuity correction** — Wilson-with-continuity-correction
  slightly conservative; ordinary Wilson usually preferred.

## Related in this repo

- `binomial-test`, `fisher-exact`, `chi-square-tests` — related
  categorical inference.
- `rates-proportions`, `credible-intervals-hpd` — nearby CI
  cousins.

## Run

```
python techniques/wilson-score-interval-proportion/python/wilson_score_interval_proportion.py
Rscript techniques/wilson-score-interval-proportion/r/wilson_score_interval_proportion.R
```

**Refs:** Wilson, E.B. "Probable inference, the law of succession, and statistical inference." *JASA*, 22(158): 209-212, 1927; Agresti, A. & Coull, B.A. "Approximate is better than exact for interval estimation of binomial proportions." *American Statistician*, 52(2): 119-126, 1998; Brown, L.D., Cai, T.T. & DasGupta, A. "Interval estimation for a binomial proportion." *Statistical Science*, 16(2): 101-133, 2001.

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
