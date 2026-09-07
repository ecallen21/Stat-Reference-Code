# Hurdle Model (Reference §7.20)

Mullahy (1986); Cragg (1971, double-hurdle). Two-part model for
zero-heavy counts (or continuous outcomes):

1. **Binary hurdle**: `P(Y > 0 | X)` — logistic / probit.
2. **Positive part**: `p(Y | Y > 0, X)` — zero-truncated Poisson /
   NegBin (counts) or log-normal (continuous).

## Contrast with zero-inflated

| Model | Zeros come from |
|---|---|
| Hurdle | ONE source; gate then count |
| Zero-inflated | TWO sources (structural + sampling) |

Hurdle keeps the two parts independent; ZIP mixes them.

## Files

- `python/hurdle_model.py` — logistic hurdle + truncated Poisson
  positive part fit by L-BFGS from scratch, plus plain Poisson
  baseline. Demo (n=3000, zero fraction 67%): hurdle β̂ ≈ (−0.82,
  1.13, −0.23) close to truth (−0.40, +1.00, −0.20); positive γ̂ ≈
  (0.46, 0.45, −0.11) close to truth (+0.40, +0.50, −0.10); plain
  Poisson mixes the two and understates the effect.
- `r/hurdle_model.R` — `pscl::hurdle`, `countreg::hurdle`, `hurdlr`,
  `glmmTMB` (R); `statsmodels.HurdleCountModel`, pymc, from-scratch
  (Python).

## When to use

- **Excess zeros with a clear "gate"** — utilisation counts (0 vs
  positive visits), engagement (0 vs > 0 sessions), insurance
  claims.
- **Two decision processes** — deciding whether to use vs how much.
- **Continuous positive part** — Cragg's double-hurdle for
  expenditure / duration.

## When NOT to use

- **Zeros arise from over-dispersion alone** — ZIP / NegBin without
  hurdle may be more parsimonious.
- **Interpretation demands "structural zero" concept** — use
  zero-inflated instead.
- **Small n / rare positives** — separation in the hurdle stage;
  use penalised (Firth) logistic.

## Assumptions & caveats

- **Independence of gate and count** — hurdle assumes them
  conditionally independent given X. If they share a latent, use
  a copula-linked hurdle or ZI model.
- **Truncated-Poisson identification** — inflated variance vs plain
  Poisson at low λ; expect NegBin extensions when overdispersed.
- **Marginal effects** are combinations of both parts (McDonald &
  Moffitt 1980 decomposition).
- **Prediction on the observed scale** = `π̂(X) · E[Y | Y > 0, X]`.

## Related in this repo

- `zero-inflated-regression` — the ZIP / ZINB alternative.
- `poisson-regression`, `negative-binomial-regression` — count
  baselines.
- `logistic-regression`, `firth-logistic` — the gate model.
- `heckman-selection` — related two-part continuous selection.

## Run

```
python techniques/hurdle-model/python/hurdle_model.py
Rscript techniques/hurdle-model/r/hurdle_model.R
```

**Refs:** Mullahy, J. "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3): 341-365, 1986; Cragg, J.G. "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica*, 39(5): 829-844, 1971.

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
