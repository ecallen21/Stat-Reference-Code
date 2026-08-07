# Isotonic Regression (Reference §5.29)

Fit a **monotone** (nondecreasing or nonincreasing) function `ŷ(x)` that minimizes weighted squared error:

```
minimize  Σ_i w_i (y_i − ŷ_i)²
subject to ŷ_1 ≤ ŷ_2 ≤ ... ≤ ŷ_n         (nondecreasing case, x sorted ascending)
```

## Pool-Adjacent-Violators Algorithm (PAVA, Ayer et al. 1955)

```
1. Sort observations by x.
2. Scan left-to-right; if consecutive blocks violate the monotone constraint
   (block_i mean > block_{i+1} mean for a nondecreasing fit), MERGE them
   into a single block with the weighted mean.
3. Continue until all blocks are monotone.
```

Output is a piecewise-constant monotone function.

## Files

- `python/isotonic_regression.py` — from-scratch PAVA with weighted-mean merging. Demo (n = 60, target `log(1+x)`): fitted RSS to truth = 1.664, **matching sklearn IsotonicRegression exactly**; fitted values are guaranteed monotone.
- `r/isotonic_regression.R` — base R `isoreg` (Robertson-Wright-Dykstra 1988 canonical implementation).

## Applications

- **Calibration of classifier probabilities** — alternative to Platt scaling; less parametric.
- **Dose-response** where monotonicity is a substantive constraint.
- **Order-restricted inference** — testing / estimating under monotone alternatives.
- **Preprocessing** for downstream methods that require monotone inputs.

## Assumptions & caveats

- **Monotonicity** must be a substantive assumption; PAVA won't discover it if the truth is non-monotone.
- **Piecewise constant** — for a smooth monotone fit, use monotone splines (`scam::scam` in R, `monotonic_slopes` in mgcv, or shape-constrained P-splines).
- **Weighted variant** (`w_i`) handles heteroscedastic data.
- **Bootstrap** for pointwise confidence bands; asymptotic distributions of PAVA fits are non-standard (n^(1/3) rate near a strict monotone region).

## Run

```
python techniques/isotonic-regression/python/isotonic_regression.py
Rscript techniques/isotonic-regression/r/isotonic_regression.R
```

**Refs:** Ayer, M. et al. "An empirical distribution function for sampling with incomplete information." *Ann. Math. Stat.* 26(4), 641–647, 1955; Barlow, R.E. et al. *Statistical Inference under Order Restrictions*, Wiley, 1972; Robertson, T., Wright, F.T. & Dykstra, R.L. *Order Restricted Statistical Inference*, Wiley, 1988.

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
