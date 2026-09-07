# Overlap Weighting -- ATO estimand (Reference §15.31)

Li, Morgan & Zaslavsky (2018). Propensity-score weighting whose target
population is the region of clinical equipoise -- units for whom
treatment assignment is genuinely uncertain (e(x) near 0.5).

## Weight schemes

| Estimand | Weight | Target |
|---|---|---|
| **ATE** | `T/e + (1−T)/(1−e)` | Full population |
| **ATT** | `T + (1−T)·e/(1−e)` | Treated population |
| **ATO** | `T·(1−e) + (1−T)·e` | Overlap population |

ATO's weights are **bounded in [0, ½]** — no extreme weights, no
truncation, and covariates are **exactly balanced** when propensity is
logistic.

## Files

- `python/overlap_weighting.py` — fits logistic propensity, computes
  ATE / ATT / ATO Hajek estimators, checks covariate balance. Demo
  (n=4000, τ=1+0.5·x₀, strong confounding): ATO estimate +1.02 vs
  overlap-weighted truth +1.00; max IPTW weight 62.3 vs max ATO weight
  0.98. Post-weighting SMD on x₀: ATE = +0.05, ATO = +0.00.
- `r/overlap_weighting.R` — `PSweight::PSweight(weight="overlap")`,
  `WeightIt::weightit(estimand="ATO")`, `cobalt::bal.tab` (R);
  scikit-learn + custom formula (Python).

## When to use

- **Extreme propensities** — near 0 or 1 make IPTW estimates unstable;
  ATO down-weights them naturally.
- **Randomised-controlled-trial mimicry** — the target is the
  population where a clinician might genuinely choose either arm.
- **Small overlap** — IPTW blows up; ATO recovers meaningful causal
  contrasts in the overlap region.

## When NOT to use

- **Population-level ATE / policy questions** — ATO answers a
  different question (the overlap sub-population is not the marginal).
- **Very strong positivity** — if e(x) ∈ [0.2, 0.8] for all x, IPTW
  and ATO agree; the extra machinery buys nothing.

## Assumptions & caveats

- **Unconfoundedness** — same as IPTW (`Y(t) ⊥ T | X`).
- **Positivity** — ATO tolerates thin overlap because it targets it,
  but a truly unmeasured / non-overlapping subgroup is still lost.
- **Estimand interpretation** — the ATO population is data-defined;
  report the weighted covariate means so readers know who it applies
  to.
- **Variance** — bootstrap for SE (analytic sandwich underestimates
  when propensity is estimated).

## Related in this repo

- `iptw`, `propensity-score-matching` — companion PS methods.
- `entropy-balancing`, `coarsened-exact-matching` — non-PS balancing
  alternatives.
- `aipw-doubly-robust` — combine ATO weights with outcome regression
  for robustness.

## Run

```
python techniques/overlap-weighting/python/overlap_weighting.py
Rscript techniques/overlap-weighting/r/overlap_weighting.R
```

**Refs:** Li, F., Morgan, K.L. & Zaslavsky, A.M. "Balancing covariates via propensity score weighting." *Journal of the American Statistical Association*, 113(521): 390-400, 2018.

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
