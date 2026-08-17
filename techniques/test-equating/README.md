# Test Equating (Reference §22.12)

Two forms `X` and `Y` designed to measure the same construct. **Equating** maps a score on `Y` to the equivalent score on `X` so examinees aren't advantaged or disadvantaged by which form they took.

## Three classical methods

- **Mean equating** — shift only:

```
y_eq = y − mean(Y) + mean(X)
```

- **Linear equating** (Tucker / Levine) — shift + scale:

```
y_eq = mean(X) + (sd(X) / sd(Y)) · (y − mean(Y))
```

- **Equipercentile equating** (Braun-Holland / Kolen-Brennan) — match percentiles:

```
y_eq = F_X⁻¹(F_Y(y))
```

Nonparametric; **kernel-smoothed** in practice to reduce sampling noise.

## Files

- `python/test_equating.py` — from-scratch mean / linear / equipercentile with a random-groups example (n = 1000 per form). Demo: for a Y-score of 60, mean = 62.6, linear = 64.0, equipercentile = 64.3.
- `r/test_equating.R` — `equate::equate` (most flexible), `kequate::keeq2` (kernel), `SNSequate` (comprehensive).

## When to use each

- **Mean** — simplest; use when only test-form difficulty differs.
- **Linear** — different means and different SDs but same shape distribution.
- **Equipercentile** — most flexible; use for large samples (n ≥ 1500 per form).
- **IRT equating** (Stocking-Lord, Haebara) — required for adaptive or common-item designs.

## Designs

- **Random groups** — each form to an independent random subsample. Simplest.
- **Single group with counterbalancing** — one sample takes both forms; controls for practice / fatigue with A-B / B-A ordering.
- **Common-item non-equivalent groups (CINEG)** — anchor items appear in both forms; groups may differ in ability. Common in operational testing.

## Assumptions & caveats

- **Same content and construct** — otherwise equating hides real differences.
- **Sample size** — equipercentile is noisy below n ≈ 1000; smoothing helps.
- **Extrapolation** — equating at the extremes is unstable; report the score range where equating is trustworthy.
- **CINEG designs** need identifying assumptions (Tucker / Levine).

## Run

```
python techniques/test-equating/python/test_equating.py
Rscript techniques/test-equating/r/test_equating.R
```

**Refs:** Kolen, M.J. & Brennan, R.L. *Test Equating, Scaling, and Linking*, 3rd ed., Springer, 2014; von Davier, A.A. (ed.) *Statistical Models for Test Equating, Scaling, and Linking*, Springer, 2011.

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
