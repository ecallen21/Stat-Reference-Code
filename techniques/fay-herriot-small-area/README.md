# Fay-Herriot Small-Area Model (Reference §27.6)

Fay & Herriot (1979). The canonical area-level linear mixed model
for **small-area estimation** — combines direct survey estimates
with covariate information via a random-effects shrinkage.

## Model

    θ_i       = x_i' · β + v_i               (true small-area mean)
    y_i       = θ_i + e_i                     (direct survey estimate)
    v_i       ~ N(0, σ_v²)                    (area random effect)
    e_i       ~ N(0, D_i)                     (design-based, D_i known)

## EBLUP

    θ̂_i = γ_i · y_i + (1 − γ_i) · x_i' · β̂
    γ_i = σ_v² / (σ_v² + D_i)

Precise areas (small `D_i`) get `γ ≈ 1` — direct estimate kept.
Noisy areas (large `D_i`) get shrunk toward the covariate model.

## Files

- `python/fay_herriot_small_area.py` — profile-likelihood MLE of
  `(β, σ_v²)` with GLS closed form + EBLUP + shrinkage weights.
  Demo (n=100, β=(2.0, 0.7), σ_v²=0.25, heterogeneous known D_i):
  β̂=(1.98, 0.81), σ̂_v² = 0.268. EBLUP MSE 0.124 beats direct 0.317
  and synthetic 0.234 — 61% MSE reduction over direct.
- `r/fay_herriot_small_area.R` — `sae::eblupFH`, `sae::mseFH` (with
  Prasad-Rao MSE), `JoSAE`, `emdi`, `BayesSAE` (R);
  `samplics.sae.eblupFayHerriot` (Python).

## When to use

- **Survey-based small-area estimates** — county-level poverty,
  small-region unemployment, sub-national health indicators.
- **Direct estimates too noisy** — few respondents per area drives
  `D_i` up.
- **You have area-level auxiliary data** — census, administrative,
  or remote-sensing covariates.

## When NOT to use

- **Unit-level auxiliary data available** — use the Battese-Harter-
  Fuller unit-level model instead (more efficient).
- **Non-continuous outcomes** — use area-level GLMM / Poisson-gamma
  small-area (see `poisson-gamma-empirical-bayes`).
- **Very few areas** (n < 20) — random-effect variance is poorly
  estimated; consider fixed-effects synthesis.

## Assumptions & caveats

- **`D_i` known** — usually taken from design-based variance
  estimates; error in `D_i` propagates.
- **Normality of `v_i`, `e_i`** — moderate robustness; use RML or
  Bayes for skewed effects.
- **Model correctness** — a mis-specified covariate model biases the
  synthetic component; check via cross-validation across areas.
- **MSE reporting** — Prasad-Rao / Jackknife / Datta-Lahiri MSE all
  account for extra uncertainty in `σ̂_v²`.

## Related in this repo

- `poisson-gamma-empirical-bayes` — count / rate variant.
- `bayesian-hierarchical-models` — full-Bayes counterpart.
- `james-stein-shrinkage`, `empirical-bayes` — theoretical roots.
- `two-stage-cluster-sampling`, `complex-survey-design` — the design
  side of survey estimation.

## Run

```
python techniques/fay-herriot-small-area/python/fay_herriot_small_area.py
Rscript techniques/fay-herriot-small-area/r/fay_herriot_small_area.R
```

**Refs:** Fay, R.E. & Herriot, R.A. "Estimates of income for small places: an application of James-Stein procedures to census data." *JASA*, 74(366): 269-277, 1979; Rao, J.N.K. & Molina, I. *Small Area Estimation*, 2nd ed., Wiley, 2015.

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
