# Competing-Risks Analysis (Reference §11.22, §11.23, §11.24, §11.25)

When more than one event type can end follow-up (death from cancer vs. death from other causes; failure by mode A vs. B), each competing risk removes subjects from being at risk of the others. **Applying 1 − KM as if only one event exists systematically OVERESTIMATES cumulative incidence** — the demo shows the naive figure at 0.75 vs. the correct 0.63.

## Aalen-Johansen CIF (§11.22)

Cumulative Incidence Function for cause k:

```
CIF_k(t)  =  ∫₀ᵗ S(u⁻) · dH_k(u)
```

where `S(u⁻)` is the overall (any-event) survival and `H_k` is the cause-k Nelson-Aalen hazard. In discrete event-time form:

```
CIF_k(t)  =  Σ_{u_j ≤ t}  Ŝ(u_j⁻) · d_k(u_j) / n(u_j)
```

## Gray's test (§11.23)

Log-rank-style test for equality of CIFs between groups. The Python file ships a **simplified 2-group approximation**; for a rigorous implementation see `cmprsk::cuminc` in R (mirrored in the R file).

## Cause-specific Cox (§11.24)

Fit a Cox model to cause k, treating other-cause events as **censored**. Interpretation: effect on the *hazard of cause k while still at risk*. Not directly translatable to CIF.

## Fine-Gray subdistribution hazard (§11.25)

Alternative parameterization: subjects who experience a competing event **remain in the risk set** (with time-decreasing censoring-KM weights). Effect on subdistribution hazard translates *directly* into effect on CIF — often the more clinically-relevant summary.

The Python implementation here is a **simplified approximation** (extended-time modified data without full IPCW weighting). Use `cmprsk::crr` in R (or a `pysurvival`/`crr` Python implementation) for production-quality Fine-Gray.

## Files

- `python/competing_risks.py` — Aalen-Johansen CIF; cause-specific Cox (via [`cox-ph`](../cox-ph)); simplified Fine-Gray; simplified Gray's test. Recovers correct CIFs on synthetic data.
- `r/competing_risks.R` — thin wrapper around the authoritative `cmprsk` package (`cuminc`, `crr`) + cause-specific `survival::coxph`.

## Assumptions

- Independent censoring conditional on covariates.
- Competing risks are **mutually exclusive** (first-event terminates follow-up).
- If risks are dependent (e.g. a covariate shifts both), Fine-Gray gives the marginal-CIF interpretation; cause-specific gives the cause-hazard interpretation. **Report both** when they might disagree.

## Run

```
python techniques/competing-risks/python/competing_risks.py
Rscript techniques/competing-risks/r/competing_risks.R
```

**Refs:** Aalen, O.O. & Johansen, S. "An empirical transition matrix for non-homogeneous Markov chains based on censored observations." *Scand. J. Stat.* 5(3), 141–150, 1978; Gray, R.J. "A class of K-sample tests for comparing the cumulative incidence of a competing risk." *Ann. Stat.* 16(3), 1141–1154, 1988; Fine, J.P. & Gray, R.J. "A proportional hazards model for the subdistribution of a competing risk." *JASA* 94(446), 496–509, 1999; Geskus, R.B. "Cause-specific cumulative incidence estimation and the Fine and Gray model under both left truncation and right censoring." *Biometrics* 67(1), 39–49, 2011.

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
