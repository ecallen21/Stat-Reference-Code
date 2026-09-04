# Nonlinear Least Squares (Reference §35.6)

Fit `y_i = f(x_i; θ) + ε_i` by minimising `Σ (y_i − f(x_i; θ))²`.

## Gauss-Newton iteration

```
J_ij = ∂f(x_i; θ) / ∂θ_j
θ ← θ + (JᵀJ)⁻¹ Jᵀ r,   r = y − f(θ).
```

## Levenberg-Marquardt

Interpolates between Gauss-Newton (fast near the minimum) and steepest
descent (safe far from it):

```
θ ← θ + (JᵀJ + λ diag(JᵀJ))⁻¹ Jᵀ r.
```

`λ` grows on failed steps, shrinks on successful ones.

## When to use

- **Any parametric non-linear model** — pharmacokinetics
  (Michaelis-Menten), dose-response (4-PL, 5-PL), growth curves,
  binding assays.
- **Analytic gradient available** — Gauss-Newton/LM converge much
  faster than derivative-free methods.

## When NOT to use

- **Very noisy data + flat likelihood** — MCMC / Bayesian inference.
- **Multiple modes** — LM finds a local minimum; use global search
  (basin-hopping, differential evolution) first.
- **Non-differentiable f** — use Nelder-Mead / DE.

## Files

- `python/nonlinear_least_squares.py` — from-scratch LM. Two demos:
  - **Michaelis-Menten** `V x / (K + x)`: `V̂ = 5.02` (true 5.0),
    `K̂ = 3.09` (true 3.0); 10 iterations.
  - **4-PL logistic** `a + (d − a) / (1 + (x/c)^b)`: all four
    parameters recovered to within 5 %; 12 iterations.
- `r/nonlinear_least_squares.R` — `nls`, `minpack.lm::nlsLM`,
  `nlme` (R); `scipy.optimize.curve_fit`, `lmfit` (Python).

## Assumptions & caveats

- **Initial values matter** — bad start → local minimum or divergence;
  use physical intuition.
- **Damping `λ`** — LM tunes automatically; Marquardt's original
  recipe is fine.
- **Standard errors** — via `(JᵀJ)⁻¹ σ̂²` on the fitted Jacobian;
  bootstrap for small n.
- **Bounds / constraints** — use `least_squares` with box bounds
  (`trf`, `dogbox`) or reparameterise.
- **Weighted NLS** — for heteroscedasticity, weight rows by `1 / σ_i²`
  (or fit variance model jointly — see `gamlss`).

## Related in this repo

- `gmm-general` — GMM generalises NLS.
- `gamlss`, `distributional-regression` — extensions when variance
  depends on `x`.
- `nonlinear-mixed-effects` (if present) — for panel data.
- `optim-quasi-newton` (adjacent) — the numerical-optimisation cousin.

## Run

```
python techniques/nonlinear-least-squares/python/nonlinear_least_squares.py
Rscript techniques/nonlinear-least-squares/r/nonlinear_least_squares.R
```

**Refs:** Levenberg, K. "A method for the solution of certain non-linear problems in least squares." *Quart Appl Math*, 1944; Marquardt, D.W. "An algorithm for least-squares estimation of nonlinear parameters." *SIAM Journal*, 1963.

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
