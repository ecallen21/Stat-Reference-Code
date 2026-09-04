# Yeo-Johnson Transformation (Reference §41.2)

Yeo & Johnson (2000). Generalisation of Box-Cox that accepts **any
real y** (positive, zero, or negative):

```
y ≥ 0 :  ((y + 1)^λ − 1) / λ           if λ ≠ 0,   log(y + 1) if λ = 0
y  < 0 : −((−y + 1)^(2 − λ) − 1) / (2 − λ) if λ ≠ 2, −log(−y + 1) if λ = 2
```

`λ` chosen by MLE on the profile log-likelihood.

## When to use

- **Skewed data with zeros or negatives** — Box-Cox is undefined
  there.
- **ML preprocessing** where an automatic power transform is
  preferred over Box-Cox.

## When NOT to use

- **Strictly positive data** — Box-Cox is more interpretable.
- **When the log has a domain meaning** — take log directly.

## Files

- `python/yeo_johnson_transformation.py` — YJ transform + MLE λ
  scan + Shapiro normality check. Demo (n=400, log-normal shifted
  by −2 so ~30 % values are negative): MLE **λ̂ = −0.10**;
  skewness **3.85 → 0.34**.
- `r/yeo_johnson_transformation.R` — `car::powerTransform`
  (`family='yjPower'`), `bestNormalize::yeojohnson` (R);
  `sklearn.PowerTransformer(method='yeo-johnson')`,
  `scipy.stats.yeojohnson` (Python).

## Assumptions & caveats

- **Normal-error assumption** underlies the MLE; robustness limited.
- **Interpretability** — like Box-Cox, coefficients are on the
  transformed scale.
- **Back-transformation** for predictions requires care; retention
  bias when marginalising a nonlinear transform.

## Related in this repo

- `box-cox-transformation` — for strictly positive data.
- `inverse-normal-transformation` — rank-based force to exact
  normality.

## Run

```
python techniques/yeo-johnson-transformation/python/yeo_johnson_transformation.py
Rscript techniques/yeo-johnson-transformation/r/yeo_johnson_transformation.R
```

**Refs:** Yeo, I.-K. & Johnson, R.A. "A new family of power transformations to improve normality or symmetry." *Biometrika*, 2000.

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
