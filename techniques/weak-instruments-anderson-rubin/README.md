# Weak-Instrument Robust Inference -- Anderson-Rubin (Reference §15.40)

Anderson & Rubin (1949); Staiger & Stock (1997); Moreira (2003).
Classical 2SLS is biased and its Wald CIs undercover when the
instrument is **weak** (first-stage F ≪ 10).

## Two components

1. **Weak-IV diagnostics.**
   - Cragg-Donald first-stage F statistic
   - Stock-Yogo critical values (5% weak-IV bias rule: F > ≈16.4
     for one instrument)
2. **Weak-IV robust tests / CIs.**
   - Anderson-Rubin: regress `Y − X·β₀` on `Z + W`; F-test that `Z`
     has no explanatory power. CI = {β₀ : test fails to reject}.
   - Kleibergen-K, Moreira's CLR, LM tests: refinements.
   - **All valid regardless of instrument strength.**

## Files

- `python/weak_instruments_anderson_rubin.py` — Cragg-Donald F +
  grid-based Anderson-Rubin CI from scratch. Demo (n=2000):
  strong IV (F=3345) gives AR CI [+0.75, +0.82] around truth +0.80;
  weak IV (F=4.4) gives an honestly wide AR CI [−5, +1.8] — the CI
  hits the search-grid boundary, showing β is effectively
  unidentified.
- `r/weak_instruments_anderson_rubin.R` — `ivmodel::ivmodel` (AR,
  CLR, LM, K), `AER::ivreg` (weak-IV F, Wu-Hausman, Sargan),
  `weakIV::` (Kleibergen-Paap) (R); `linearmodels.iv.IV2SLS/IVGMM`
  (Python).

## When to use

- **Any IV analysis** — report the first-stage F.
- **First-stage F < 10** — do not rely on 2SLS Wald CIs; use AR / CLR.
- **Just-identified models** — AR is exact under normality; CLR is
  point-optimal under Moreira's asymptotics.

## When NOT to use

- **First-stage F ≫ 10 and one instrument** — 2SLS Wald / delta-method
  CIs are fine; the AR CI reduces to something similar.
- **Point identification when β is not the object of interest** —
  for LATE with heterogeneous compliers, LATE-robust methods apply.

## Assumptions & caveats

- **Exogeneity of Z** — AR is robust to weak Z but not to
  invalid Z.
- **Homoskedasticity** — the basic AR F test needs it; use
  heteroskedastic-robust AR (Kleibergen-Paap rk statistic) otherwise.
- **CI can be empty** — under a truly wrong H₀ range the AR CI is
  empty; extend the grid.
- **CI can be the whole real line** — indicating β is unidentified;
  do not report a point estimate.

## Related in this repo

- `iv-2sls` — the classical 2SLS estimator.
- `principal-stratification-cace` — LATE via IV under monotonicity.
- `manski-bounds` — partial identification when even AR is
  uninformative.
- `mendelian-randomization` — genetic IVs; weak-IV bias critical.

## Run

```
python techniques/weak-instruments-anderson-rubin/python/weak_instruments_anderson_rubin.py
Rscript techniques/weak-instruments-anderson-rubin/r/weak_instruments_anderson_rubin.R
```

**Refs:** Anderson, T.W. & Rubin, H. "Estimation of the parameters of a single equation in a complete system of stochastic equations." *AMS*, 20: 46-63, 1949; Staiger, D. & Stock, J.H. "Instrumental variables regression with weak instruments." *Econometrica*, 65(3): 557-586, 1997; Moreira, M.J. "A conditional likelihood ratio test for structural models." *Econometrica*, 71(4): 1027-1048, 2003.

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
