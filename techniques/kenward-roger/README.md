# Kenward–Roger / Satterthwaite denominator df for LMM contrasts (Reference §12.17)

Small-sample LMM inference is broken if you use z or a Wald t with residual df:

- REML estimates variance components **with uncertainty** — pretending they're known makes the effective sample size too big.
- The design-effective df is neither `n − p` (asymptotic) nor `n_clusters − p` (fully conservative). It's somewhere in between.

## Two standard corrections

- **Kenward–Roger (KR)** (1997): adjusts both `Cov(β̂)` and the df of a Wald test on a linear contrast `L'β`. Uses derivatives of `Cov(β̂)` w.r.t. variance components. Most accurate; involved to implement.
- **Satterthwaite**: approximates df by matching first two moments of the test statistic to a scaled χ². Simpler; widely used (`lmerTest::contest`).

## This file

Ships a **simplified Satterthwaite-style** t-test on a linear contrast — enough to demonstrate the small-sample-df idea without the full KR machinery. For production use:

- **R**: `pbkrtest::KRmodcomp` (true KR) or `lmerTest::lmer` (Satterthwaite via lme4 hooks).
- **Python**: no direct KR/Satterthwaite in statsmodels; the community workaround is to call R via `rpy2`.

## Files

- `python/kenward_roger.py` — Satterthwaite-style contrast test on top of [`linear-mixed-models`](../linear-mixed-models)'s `fit_lmm`. Uses `df = n_clusters − p` as a safe approximation for balanced designs.
- `r/kenward_roger.R` — pointers to `pbkrtest` (KR) and `lmerTest` (Satterthwaite).

## Assumptions

- LMM correctly specified (fixed and random effects).
- Balanced or near-balanced design for the simple df approximation shipped here.
- Small n / few clusters (that's when the correction matters — for n_clusters ≥ 30 the standard Wald z is fine).

## Run

```
python techniques/kenward-roger/python/kenward_roger.py
Rscript techniques/kenward-roger/r/kenward_roger.R
```

**Refs:** Kenward, M.G. & Roger, J.H. "Small sample inference for fixed effects from restricted maximum likelihood." *Biometrics* 53(3), 983–997, 1997; Satterthwaite, F.E. "An approximate distribution of estimates of variance components." *Biometrics Bull.* 2(6), 110–114, 1946; Kuznetsova, A., Brockhoff, P.B. & Christensen, R.H.B. "lmerTest package: tests in linear mixed effects models." *J. Stat. Soft.* 82(13), 1–26, 2017; Halekoh, U. & Højsgaard, S. "A Kenward-Roger approximation and parametric bootstrap methods for tests in linear mixed models — the R package pbkrtest." *J. Stat. Soft.* 59(9), 1–32, 2014.

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
