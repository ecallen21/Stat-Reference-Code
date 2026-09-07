# Bühlmann-Straub Credibility (Reference §24.21)

Bühlmann & Straub (1970). Extension of Bühlmann's model that allows
**group-specific exposures / weights** `w_g`:

    Var(X | g) = σ² / w_g

    Z_g = w_g / (w_g + K),   K = σ² / τ²
    P_g = Z_g · X̄_g + (1 − Z_g) · μ

Weighted group means; small-exposure groups get shrunk more toward
the weighted collective mean.

## Files

- `python/buhlmann_straub_credibility.py` — B-S estimator with
  weighted grand mean, MoM `σ²`, `τ²`, per-group `Z_g`, and
  shrunk prediction from scratch. Demo (G=40, exposures skewed low
  ⇒ Z ranges 0.02-0.69): raw-mean MSE 0.389 vs B-S MSE 0.028 → 93 %
  reduction.
- `r/buhlmann_straub_credibility.R` — `actuar::cm`, `ChainLadder`,
  `lme4::lmer weighted` BLUP (R); `chainladder`, from-scratch
  (Python).

## When to use

- **Actuarial rate-making with exposure** — policy-year weights,
  hospital patient-days, group-life sums-insured.
- **Any hierarchical setting with per-group exposure**.
- **Chain-ladder + credibility** — Cape Cod uses B-S-style
  weighting.

## When NOT to use

- **Equal / no exposure information** — plain Bühlmann suffices.
- **Very heavy tails** — extend to distribution-free / robust
  credibility.
- **Cross-classified structure** — use hierarchical credibility
  (Jewell 1975, Hachemeister 1975).

## Assumptions & caveats

- **Correct exposure interpretation** — `w_g` must scale variance
  inversely; otherwise MoM estimates are biased.
- **`τ²` may go negative** in noisy data; clip to 0 (all
  shrinkage) as a practical fix.
- **BLUP equivalence** to a random-intercept linear mixed model
  with per-obs weights.
- **Report Z per group** so consumers see who was shrunk.

## Related in this repo

- `buhlmann-credibility` — the unweighted precursor.
- `fay-herriot-small-area`, `poisson-gamma-empirical-bayes` —
  small-area cousins.
- `linear-mixed-models`, `james-stein-shrinkage`,
  `bayesian-hierarchical-models` — shrinkage cousins.

## Run

```
python techniques/buhlmann-straub-credibility/python/buhlmann_straub_credibility.py
Rscript techniques/buhlmann-straub-credibility/r/buhlmann_straub_credibility.R
```

**Refs:** Bühlmann, H. & Straub, E. "Glaubwürdigkeit für Schadensätze." *Bulletin of the Swiss Association of Actuaries*, 70(1): 111-133, 1970; Bühlmann, H. & Gisler, A. *A Course in Credibility Theory and its Applications*, Springer, 2005.

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
