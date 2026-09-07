# Chow + QLR Structural-Break Test (Reference §12.16)

Chow (1960); Andrews (1993, QLR / sup-F). Test that regression
coefficients are the same before and after a candidate break `τ`:

    H₀: β₁ = β₂ = β_pool

## Chow F-statistic (known τ)

    F = ((SSR_pool − SSR₁ − SSR₂) / k)
        / ((SSR₁ + SSR₂) / (n − 2k))    ~   F(k, n − 2k)

## QLR sup-F (unknown τ)

Compute Chow F over `τ ∈ [0.15n, 0.85n]`; use Andrews (1993)
critical values (not standard F).

## Files

- `python/chow_test_structural_break.py` — Chow F + QLR sup-F over
  a trimmed grid from scratch. Demo (n=200, break at t=100 in
  intercept): stable series F=0.41, p=0.66 (correctly non-
  significant); with break F=336, p≈0; QLR picks τ*=100 exactly.
- `r/chow_test_structural_break.R` — `strucchange::sctest /
  breakpoints`, `lmtest::sctest` (R); `statsmodels.stats.
  diagnostic.breaks_cusumolsresid`, `ruptures` (Python).

## When to use

- **Time-series regression** — pre / post policy change, before /
  after regime shift.
- **Testing exogenous break at known date** — Chow with fixed τ.
- **Unknown break search** — QLR sup-F with Andrews critical
  values.
- **Multiple structural breaks** — Bai-Perron (strucchange).

## When NOT to use

- **Cross-section without ordering** — no time / order axis to
  break on.
- **Non-linear or unit-root series** — sup-F distribution changes;
  use bootstrap or specialised unit-root break tests.
- **Very small sub-samples** — F approximation degrades when either
  side has few observations.

## Assumptions & caveats

- **Same variance both regimes** — Chow assumes it; Chow-2 relaxes.
- **Independent errors** — HAC / cluster robust versions available
  in `strucchange`.
- **QLR critical values** are Andrews-1993 tabulated, NOT standard F
  — using standard F over-rejects.
- **Trimming** — 15 % each side is standard to keep sub-sample
  power.

## Related in this repo

- `change-point-detection` — non-parametric alternative.
- `event-study`, `staggered-did` — cousin econometric event
  designs.
- `sequential-analysis`, `always-valid-inference` — sequential
  monitoring cousins.
- `regime-switching-markov` — model-based regime approach.

## Run

```
python techniques/chow-test-structural-break/python/chow_test_structural_break.py
Rscript techniques/chow-test-structural-break/r/chow_test_structural_break.R
```

**Refs:** Chow, G.C. "Tests of equality between sets of coefficients in two linear regressions." *Econometrica*, 28(3): 591-605, 1960; Andrews, D.W.K. "Tests for parameter instability and structural change with unknown change point." *Econometrica*, 61(4): 821-856, 1993.

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
