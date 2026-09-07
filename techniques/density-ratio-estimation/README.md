# Density Ratio Estimation (Reference §46.18)

Sugiyama, Suzuki & Kanamori (2012). Estimate `r(x) = p_num(x) /
p_den(x)` **directly** — no separate density estimates.

## Methods

| Method | Objective |
|---|---|
| **KLIEP** | `max Σ log r(x_num)` s.t. `E_den[r] = 1` |
| **uLSIF** | `min ½ E_den[r²] − E_num[r]` (unconstrained LS) |
| **RuLSIF** | relative uLSIF for robustness at large ratios |
| **Classifier (LR / NCE)** | `r̂(x) = p̂₁/(1−p̂₁) · n_den/n_num` from num-vs-den classifier |

## Applications

- **Covariate-shift adaptation** — importance-weight the training
  loss.
- **Change-point / drift detection** — ratio flags shifts.
- **Two-sample testing** — energy / kernel tests via ratios.
- **Off-policy evaluation** — importance sampling for RL.
- **Mutual-information / KL estimation** — plug-in via density
  ratios.

## Files

- `python/density_ratio_estimation.py` — uLSIF (Gaussian basis,
  Tikhonov-regularised least squares) + logistic-regression
  classifier ratio from scratch. Demo (p_num = N(0,1), p_den =
  N(1, 1.5), n=1000 each): uLSIF closely tracks the analytic ratio
  (2.186 vs 2.213 at x=−1, 2.014 vs 1.873 at x=0); classifier ratio
  overshoots at the tails (extrapolation) as expected.
- `r/density_ratio_estimation.R` — `densratio` (Sugiyama's uLSIF /
  KLIEP / RuLSIF) (R); `densratio` (Python port), scikit-learn
  odds-trick (Python).

## When to use

- **You need the ratio, not the densities** — covariate shift,
  drift, IS weights.
- **Both `p_num` and `p_den` are hard to estimate** — but ratio
  is smoother than either.
- **Two-sample discrimination** — LR-based ratios double as
  classifiers.

## When NOT to use

- **You need the densities themselves** — use KDE or a parametric
  fit.
- **Extreme ratios** — unbounded r blows up estimator variance;
  use RuLSIF or truncate.
- **Very high dimensions** — kernel-based methods struggle; deep
  DRE (via logistic classifier or NCE) scales better.

## Assumptions & caveats

- **Overlap** — `p_num` support ⊂ `p_den` support (positivity).
  Without it, r is undefined.
- **Bandwidth / regularisation** must be cross-validated.
- **Bias** — classifier-based ratio is consistent only if the
  classifier is well-calibrated; use Platt scaling.
- **Effective sample size** of importance weights — monitor
  `1 / Σ w²` for pathological weights.

## Related in this repo

- `covariate-shift-adaptation`, `data-drift-detection`,
  `concept-drift-adwin` — canonical applications.
- `kl-divergence`, `f-divergences`, `optimal-transport-wasserstein`
  — distributional-distance cousins.
- `importance-sampling`, `iptw` — weighting-based inference.
- `energy-based-models`, `score-matching` — related unnormalised-
  density estimation.

## Run

```
python techniques/density-ratio-estimation/python/density_ratio_estimation.py
Rscript techniques/density-ratio-estimation/r/density_ratio_estimation.R
```

**Refs:** Sugiyama, M., Suzuki, T. & Kanamori, T. *Density Ratio Estimation in Machine Learning*, Cambridge, 2012; Kanamori, T., Hido, S. & Sugiyama, M. "A least-squares approach to direct importance estimation." *JMLR*, 10: 1391-1445, 2009.

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
