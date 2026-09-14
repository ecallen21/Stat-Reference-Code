# Huber M-Estimator (Reference §47.382)

Huber (1964). Foundational M-estimator with loss:

```
ρ(u) = ½ u²                    if |u| ≤ k
ρ(u) = k |u| − ½ k²             otherwise
```

Influence function `ψ(u) = min(k, max(−k, u))` is BOUNDED, so
a single outlier cannot dominate. Combines OLS efficiency at
Gaussian (95 % at `k = 1.345 σ`) with linear-tail robustness
against moderate outliers. The archetypal robust regression.

## Files

- `python/huber_m_estimator.py` — n=200, p=4 regression with
  15 % wild outliers. Iteratively-reweighted LS with MAD
  scale reaches ‖β − β_true‖ = 0.077; OLS is at 0.43 (5.6×
  worse); `sklearn.linear_model.HuberRegressor` matches at
  0.085.
- `r/huber_m_estimator.R` — `MASS::rlm(method='M',
  psi=psi.huber)`, `robustbase::lmrob`, `robust::lmRob` (R);
  `sklearn.HuberRegressor`, `statsmodels.RLM`, from-scratch
  IRLS (Python).

## When to use

- **Regression with mild-to-moderate contamination** — up to
  ~ 15-20 % outliers.
- **When you want asymptotic normality and inference** —
  M-estimators have standard sandwich variance formulas.
- **Combined with S-scale** — MM-estimators get 50 %
  breakdown + 95 % Gaussian efficiency (see mm-estimators-robust).

## When NOT to use

- **Very high contamination (> 30 %)** — use LTS / LMS /
  MCD-based initialisation.
- **Leverage outliers in X** — Huber M is NOT robust to
  outliers in the design matrix; use bounded-influence /
  MM / LTS.
- **Non-continuous data** — logistic regression needs its
  own robust variants (Croux-Haesbroeck).

## Assumptions & caveats

- **Tuning constant k** — `k = 1.345` for 95 %
  Gaussian-efficiency; smaller k ⇒ more robust, less
  efficient.
- **Scale estimation** — MAD (`1.4826 × median(|r − median|)`)
  standard; use jointly with `k` or Tukey's proposal 2.
- **IRLS convergence** — typically < 20 iterations; can
  cycle on hard problems; use `robustbase::lmrob` which
  handles pathological cases.
- **Breakdown point** — Huber M has BP = 0 in X direction
  (leverage), 1/(1 + √p) in Y direction (Rousseeuw's result).
- **Robust variance-covariance** — Huber sandwich estimator
  for standard errors.

## Related in this repo

- `mm-estimators-robust`, `tukey-biweight-m-estimator` —
  higher-breakdown alternatives.
- `least-trimmed-squares`, `mcd-robust-covariance`,
  `hampel-identifier`, `winsorization` — robust neighbours.
- `robust-regression`, `sandwich-robust-se` — regression /
  inference cousins.
- `quantile-regression`, `theil-sen-slope` — quantile /
  rank-based robust alternatives.

## Run

```
python techniques/huber-m-estimator/python/huber_m_estimator.py
Rscript techniques/huber-m-estimator/r/huber_m_estimator.R
```

**Refs:** Huber, P.J. "Robust estimation of a location parameter." *Ann. Math. Statist.*, 35: 73-101, 1964; Huber, P.J. and Ronchetti, E.M. *Robust Statistics*, 2nd ed., Wiley, 2009.

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
