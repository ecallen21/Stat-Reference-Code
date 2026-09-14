# Tukey Biweight M-Estimator (Reference §47.383)

Beaton & Tukey (1974). Redescending M-estimator with loss:

```
ρ(u) = (c²/6) · (1 − (1 − (u/c)²)³)     if |u| ≤ c
ρ(u) = c²/6                              otherwise
```

Influence function `ψ(u) = u · (1 − (u/c)²)²` REDESCENDS to
zero for `|u| > c` — so extreme outliers get ZERO influence,
unlike Huber where they still contribute linearly. Standard
tuning `c = 4.685 σ` for 95 % Gaussian efficiency. Non-convex,
so a good warm start (Huber-like, LTS, or MM) is essential.

## Files

- `python/tukey_biweight_m_estimator.py` — n=200, p=4
  regression with 20 % wild outliers (uniform in ±40 units).
  OLS breaks (‖β − β_true‖ = 2.18). A Huber-like warm start
  gets to 0.10; refined Tukey biweight reaches 0.08 — 27×
  tighter than OLS.
- `r/tukey_biweight_m_estimator.R` —
  `MASS::rlm(psi='psi.bisquare')`,
  `robustbase::lmrob(method='MM')` (R);
  `statsmodels.RLM(M=TukeyBiweight())`, from-scratch (Python).

## When to use

- **Regression with heavy contamination** — redescending
  ψ ignores obvious outliers.
- **Combined with S / LTS scale start** — MM-estimators give
  50 % breakdown + 95 % Gaussian efficiency; `robustbase::lmrob`
  is the practical reference.
- **When you want fully bounded-influence estimates**.

## When NOT to use

- **Bad initialisation** — non-convex loss ⇒ multiple local
  minima; ALWAYS start from a high-breakdown estimate (LTS
  or S-scale).
- **Leverage outliers** — bounded influence only in Y; use
  Tukey + bounded leverage (Tau or S-estimator with
  Mallows-type weights).
- **Very few observations** — asymptotic normality relies
  on n large enough that outliers stay a bounded fraction.

## Assumptions & caveats

- **Tuning c** — `c = 4.685` for 95 % Gaussian efficiency;
  `c = 3.44` gives 85 % efficiency but higher breakdown-like
  behaviour.
- **Warm start** — LTS / S / MAD-based / Huber first, then
  Tukey; SUB-OPTIMAL local minima otherwise.
- **Scale s must be robust** — MAD or S-scale; ML scale is
  not robust.
- **Convergence** — IRLS on non-convex; monitor objective;
  cycle detection may be needed.
- **Standard errors** — sandwich estimator using the
  M-estimator asymptotics; bootstrap alternatives.

## Related in this repo

- `huber-m-estimator`, `mm-estimators-robust`,
  `least-trimmed-squares`, `mcd-robust-covariance`,
  `hampel-identifier` — robust neighbours.
- `robust-regression`, `robust-location-scale`,
  `sandwich-robust-se` — inference for robust estimators.
- `winsorization`, `trimmed-mean` — one-off robust
  transformations.

## Run

```
python techniques/tukey-biweight-m-estimator/python/tukey_biweight_m_estimator.py
Rscript techniques/tukey-biweight-m-estimator/r/tukey_biweight_m_estimator.R
```

**Refs:** Beaton, A.E. and Tukey, J.W. "The fitting of power series, meaning polynomials, illustrated on band-spectroscopic data." *Technometrics*, 16: 147-185, 1974.

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
