# Gauss-Newton / Levenberg-Marquardt (Reference §47.321)

Levenberg (1944); Marquardt (1963). Nonlinear least squares
`min Σ r_i(x)²` iterates:

```
Gauss-Newton:        (JᵀJ) dx = −Jᵀ r     (J = Jacobian of r)
Levenberg-Marquardt: (JᵀJ + λ I) dx = −Jᵀ r
```

LM interpolates between Gauss-Newton (λ → 0) and steepest
descent (λ → ∞); λ is adapted based on step quality.

## Files

- `python/gauss_newton_lm_nlls.py` — LM with adaptive damping
  on exponential regression `y = a exp(−b x) + c`. Estimates
  (a, b, c) = (3.00, 0.70, 0.50) recovered from true (3.0,
  0.7, 0.5) in ~10 iterations; RSS drops from ~50 to ~0.23.
- `r/gauss_newton_lm_nlls.R` — `stats::nls`,
  `minpack.lm::nlsLM` (R);
  `scipy.optimize.least_squares(method='lm')`, `lmfit`,
  from-scratch (Python).

## When to use

- **Nonlinear least squares** — chemistry rate constants,
  pharmacokinetics, geophysics inversion.
- **Bundle adjustment** in computer vision.
- **Model calibration** with mildly nonlinear structure.

## When NOT to use

- **Large problems** — JᵀJ is dense; use L-BFGS or
  trust-region-CG.
- **Non-least-squares objectives** — use L-BFGS / gradient
  descent.
- **Very rough loss surfaces** — LM can stall in local minima;
  add multistart.

## Assumptions & caveats

- **Jacobian availability** — analytic Jacobian preferred;
  finite differences slow.
- **λ update rule** — accept step → shrink λ; reject →
  grow. Marquardt used ×10 / ÷10; libraries commonly use ×2 / ÷2.
- **Rank-deficient J** — regularise (λI) prevents singular
  systems.
- **Trust-region variants** — `dogbox`, `trf` in SciPy are
  more robust for bound constraints.

## Related in this repo

- `nonlinear-least-squares` — the general framework.
- `lbfgs-quasi-newton` — smoother-objective alternative.
- `conjugate-gradient-cg` — inner solver for large J^T J.
- `iv-2sls`, `sur-regression` — related two-stage / joint
  estimation.

## Run

```
python techniques/gauss-newton-lm-nlls/python/gauss_newton_lm_nlls.py
Rscript techniques/gauss-newton-lm-nlls/r/gauss_newton_lm_nlls.R
```

**Refs:** Levenberg, K. "A method for the solution of certain nonlinear problems in least squares." *Q. Appl. Math.*, 2(2): 164-168, 1944; Marquardt, D.W. "An algorithm for least-squares estimation of nonlinear parameters." *SIAM J.*, 11(2): 431-441, 1963.

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
