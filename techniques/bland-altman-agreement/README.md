# Bland-Altman Agreement Analysis (Reference §13.16)

Bland & Altman (1986). Compares two continuous measurement methods
`A` and `B` on the same units:

    bias                   = mean(A − B)
    95% limits of agreement = bias ± 1.96 · sd(A − B)
    proportional bias       = regression slope of (A − B) on (A + B)/2

**Bland-Altman plot**: y = A − B, x = (A + B)/2, with reference lines
at bias and LoA. Reveals mean-dependent bias / heteroskedasticity.

Distinct from correlation / R² — two methods can correlate 0.99 yet
disagree substantially (constant offset).

## Files

- `python/bland_altman_agreement.py` — from-scratch bias, LoA,
  Bland-Altman-1999 confidence intervals for bias & LoA, plus
  proportional-bias regression. Demo (n=200, method B = 0.98·A +
  1.5 + noise): bias +0.77 (95% CI 0.34, 1.20), LoA ±6.1, slope
  +0.011 (small proportional bias). `corr(A, B) = 0.977` HIDES the
  ±6 unit disagreement — motivating the analysis.
- `r/bland_altman_agreement.R` — `blandr`, `BlandAltmanLeh`,
  `agRee`, `MethComp` (R); `pyCompare`, from-scratch (Python).

## When to use

- **Method comparison** — clinical devices, assays, wearables.
- **Deming / Passing-Bablok complement** — combine slope-based
  method comparison with BA agreement.
- **Repeatability / intra-rater** — same instrument twice.

## When NOT to use

- **Categorical outcomes** — use kappa / weighted kappa.
- **When you need slope + intercept for measurement conversion** —
  use Deming regression or Passing-Bablok.
- **Very small n** — LoA CIs are wide; ≥60 recommended (Bland &
  Altman 1999).

## Assumptions & caveats

- **Normally distributed differences** — check with Shapiro or Q-Q;
  transform if skewed (log for multiplicative methods).
- **Constant variance across the range** — visible in the BA plot;
  if not, model heteroskedasticity or split by range.
- **Report both bias AND LoA** with CIs — a small bias with huge LoA
  means individuals still disagree.
- **Clinical vs statistical** significance — decide up-front what LoA
  width is acceptable clinically.

## Related in this repo

- `agreement-beyond-kappa` — Gwet AC / Krippendorff for categorical.
- `intraclass-correlation`, `cohens-kappa` — related agreement
  metrics.
- `concordance-correlation` — Lin's concordance for continuous.
- `measurement-error-models` — model-based measurement-error
  approach.

## Run

```
python techniques/bland-altman-agreement/python/bland_altman_agreement.py
Rscript techniques/bland-altman-agreement/r/bland_altman_agreement.R
```

**Refs:** Bland, J.M. & Altman, D.G. "Statistical methods for assessing agreement between two methods of clinical measurement." *Lancet*, 1(8476): 307-310, 1986; Bland, J.M. & Altman, D.G. "Measuring agreement in method comparison studies." *Statistical Methods in Medical Research*, 8(2): 135-160, 1999.

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
