# Winsorization and Truncation (Reference §41.5)

Wilcox (2022). Two ways to reduce the influence of extreme
observations:

- **Winsorization** — replace values below the `q_lo` percentile
  with the `q_lo` value; similarly at the top. Sample size
  preserved.
- **Truncation / trimming** — remove values outside `[q_lo, q_hi]`.
  Sample size reduced.

Winsorization has smaller variance than trimming at the cost of a
small bias; symmetric 10-20 % Winsorization is common for robust
mean estimation.

## When to use

- **Robust point estimation** of the mean when outliers are known
  to be measurement errors, not real tail events.
- **Preprocessing** before OLS if you want to keep n intact.

## When NOT to use

- **Tail behaviour is the target** — Winsorizing crushes exactly
  the events you want to study (use EVT or quantile regression
  instead).
- **Genuine outliers** deserve investigation, not automatic capping.

## Files

- `python/winsorization.py` — symmetric Winsor + trim at 5 / 10 /
  20 %. Demo (n=200 normal + 5 extreme outliers): raw mean 55.7,
  sd 70.8; Winsor-5 % **mean 50.44, sd 9.28**; trim-5 % **mean
  50.41, sd 8.25**, n=180.
- `r/winsorization.R` — `DescTools::Winsorize`, `robustHD`,
  `psych::winsor` (R); `scipy.stats.mstats.winsorize`, `pandas.clip`
  (Python).

## Assumptions & caveats

- **Percentile choice** should be justified — commonly 5 %, 10 %,
  or 20 %.
- **Symmetric vs asymmetric** — one-sided caps may make sense for
  bounded-below variables (income, length of stay).
- **Downstream SEs** — Winsorized SD is biased downward as a
  variance estimate; use a bootstrap SE for the Winsorized mean.
- **Winsorizing before inference** is a form of data-dependent
  transformation; report the cutoffs.

## Related in this repo

- `robust-regression` (if present) — robust methods that avoid
  Winsorizing.
- `standardization-scaling` — robust scalers are the multi-feature
  version.

## Run

```
python techniques/winsorization/python/winsorization.py
Rscript techniques/winsorization/r/winsorization.R
```

**Refs:** Wilcox, R.R. *Introduction to Robust Estimation and Hypothesis Testing*, 5th ed., Academic Press, 2022.

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
