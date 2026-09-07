# 2PL / 3PL IRT (Reference §20.4, §20.5)

Birnbaum (1968), Baker & Kim (2004). Item Response Theory
extensions of the Rasch (1PL) model with per-item discrimination
and optional guessing.

## Models

| Model | `P(y_ij = 1 | θ_i)` |
|---|---|
| **1PL** (Rasch) | `σ(θ − b_j)` |
| **2PL** | `σ(a_j · (θ − b_j))` |
| **3PL** | `c_j + (1 − c_j) · 2PL_ij` |

- `a_j` — item discrimination.
- `b_j` — item difficulty.
- `c_j` — pseudo-guessing (lower asymptote).

Estimated by **marginal MLE** — integrate `θ ~ N(0, 1)` out via
Gauss-Hermite quadrature, then optimise over item parameters.

## When to use

- **Educational testing** — item bank calibration.
- **Psychometric scales** — clinical / patient-reported outcome
  measures.
- **Adaptive testing** — 2PL / 3PL support computer-adaptive test
  algorithms.

## When NOT to use

- **Small n / few items** — 3PL is weakly identified; stick to
  1PL or 2PL.
- **Polytomous items** — use GRM / GPCM / PCM instead.

## Files

- `python/irt_2pl_3pl.py` — marginal MLE via GH quadrature +
  L-BFGS-B (custom). Demo (n=800, J=6, guessing c=0.15): 2PL b
  parameters recovered to ~0.2 of truth; 3PL c-mean estimated at
  0.225 (truth 0.15); 3PL a values noisy — the classical 3PL
  identification issue with modest n.
- `r/irt_2pl_3pl.R` — `ltm::ltm`/`rasch`, `mirt`, `TAM` (R);
  `girth`, `py-irt`, custom (Python).

## Assumptions & caveats

- **Unidimensionality** — one latent θ underlies responses; test
  before fitting.
- **Local independence** — item responses conditional on θ are
  independent.
- **3PL identification** — c parameter is weakly identified without
  many items and low-ability respondents; use informative priors
  (Bayesian IRT).
- **Report item info curves** — max information around b for 2PL;
  shifts under 3PL guessing.

## Related in this repo

- `rasch-model` (1PL), `mirt-multidimensional-irt` — companions.
- `latent-class-analysis`, `latent-profile-analysis` — categorical
  latent alternatives.

## Run

```
python techniques/irt-2pl-3pl/python/irt_2pl_3pl.py
Rscript techniques/irt-2pl-3pl/r/irt_2pl_3pl.R
```

**Refs:** Birnbaum, A. "Some latent trait models and their use in inferring an examinee's ability." In *Statistical Theories of Mental Test Scores*, 1968; Baker, F.B. & Kim, S.-H. *Item Response Theory: Parameter Estimation Techniques*, 2nd ed., CRC, 2004.

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
