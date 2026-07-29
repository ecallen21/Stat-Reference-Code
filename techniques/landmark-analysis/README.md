# Landmark Analysis (Reference §11.24)

Survival analysis when the exposure of interest is **time-varying** — transplant, response-to-treatment, biomarker turning positive — and can only be received by subjects who have survived long enough to receive it.

## The bias it fixes

Comparing **ever-exposed** vs **never-exposed** as fixed baseline groups (or defining exposure at time zero based on future information) produces **immortal-time bias**: exposed subjects are guaranteed to have survived at least until their exposure time, artificially inflating their apparent survival benefit.

## The landmark method (Anderson et al. 1983)

1. Pick a landmark time `t*`.
2. Restrict the analysis to subjects still alive and event-free at `t*`.
3. Classify each surviving subject by exposure status **at `t*`** (had exposure by then vs not).
4. Compare survival from `t*` forward using standard KM / log-rank / Cox with the fixed landmark-time exposure covariate.

## Trade-offs

- **+** Eliminates immortal-time bias — the exposed and unexposed groups both must have survived to `t*`.
- **+** Uses standard survival tools once the cut is made.
- **−** Discards events and information before `t*`.
- **−** Ignores exposures that occur after `t*`.
- **−** Sensitive to the choice of `t*` — always run a **super-landmark** / **dynamic landmark** sensitivity analysis over multiple `t*`.

## Alternatives

- **Time-dependent Cox** (`survival::coxph` with `tt()` or `(start, stop]` data) uses the full follow-up; both approaches address immortal-time bias but landmark is simpler to communicate.
- **Cloning + inverse-probability censoring weighting** for target-trial emulation.

## Files

- `python/landmark_analysis.py` — from-scratch KM + log-rank + landmark cut + super-landmark sensitivity. Demo shows naive ever/never yielding a spurious p ≈ 2 × 10⁻⁵ while landmark p-values across `t* ∈ {1, 2, 3, 4}` reflect the null.
- `r/landmark_analysis.R` — `survival::survdiff` after the landmark subset.

## Assumptions

- Exposure-status classification at `t*` is knowable and accurate.
- No dependent censoring; standard survival assumptions apply after the cut.
- Enough events after `t*` to power the comparison.

## Run

```
python techniques/landmark-analysis/python/landmark_analysis.py
Rscript techniques/landmark-analysis/r/landmark_analysis.R
```

**Refs:** Anderson, J.R., Cain, K.C. & Gelber, R.D. "Analysis of survival by tumor response." *J. Clin. Oncol.* 1(11), 710–719, 1983; Van Houwelingen, H.C. "Dynamic prediction by landmarking in event history analysis." *Scand. J. Stat.* 34(1), 70–85, 2007.

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
