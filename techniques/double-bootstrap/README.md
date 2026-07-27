# Double (Iterated) Bootstrap for CI Calibration (Reference §10.11)

The plain single-level bootstrap gives a CI whose **actual** coverage may differ from the nominal level in finite samples. The **double bootstrap** uses a second level of resampling to *calibrate* the nominal level so that the calibrated CI has the intended coverage.

## Algorithm

1. Draw `B₁` first-level bootstrap replicates `x*` from the data.
2. For each `x*`:
   - Compute `θ*`.
   - Draw `B₂` **second-level** bootstrap replicates `x**` from `x*`.
   - Build the plain percentile CI on `θ**` at nominal level `α₀`.
   - Record whether the original `θ̂` (proxy for the "true" parameter) falls inside.
3. Empirical inner-coverage `= mean(covered)`; call `α_emp = 1 − emp_cov`.
4. **One-step Beran (1987) calibration**:
   ```
   α_calibrated = α₀² / α_emp
   ```
   which is smaller than α₀ (wider CI) when α_emp > α₀ (undercoverage) and larger (narrower CI) when the plain CI over-covers.
5. Report the plain and calibrated outer-percentile CIs.

## Cost

`B₁ · B₂` resamples — quadratic. Use small `B₁, B₂` (a few hundred each) for demonstration; production runs pick `B₁, B₂ ∈ [500, 1000]`.

## When to use

- Small samples where you suspect the plain percentile CI has bad coverage.
- Complicated statistics where you don't want to code BCa's jackknife acceleration.
- Sanity check on any bootstrap CI you're about to report.

BCa (see [`bca-bootstrap`](../bca-bootstrap)) is a cheaper *first-order* correction; the double bootstrap gives *second-order* correction at the cost of an inner loop.

## Files

- `python/double_bootstrap.py` — from-scratch with the one-step Beran calibration; reports plain vs. calibrated percentile CIs.
- `r/double_bootstrap.R` — from-scratch version.

## Assumptions

- Same as any bootstrap (IID observations; enough distinct values).
- The one-step calibration is a linear approximation to the coverage function — for wildly miscalibrated cases, a full iterated calibration (search for the α that yields exact coverage) is more robust; not implemented.

## Run

```
python techniques/double-bootstrap/python/double_bootstrap.py
Rscript techniques/double-bootstrap/r/double_bootstrap.R
```

**Refs:** Beran, R. "Prepivoting to reduce level error of confidence sets." *Biometrika* 74(3), 457–468, 1987; Hall, P. "On the bootstrap and confidence intervals." *Ann. Stat.* 14(4), 1431–1452, 1986; Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap*, Chapman & Hall, 1993 (Ch. 25).

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
