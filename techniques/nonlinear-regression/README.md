# Nonlinear Least Squares Regression (Reference §5.13)

Model:

```
y_i = f(x_i, θ) + ε_i,     ε_i ~ N(0, σ²)
θ̂ = argmin  Σ_i (y_i − f(x_i, θ))²
```

Contrast with linear regression: `f` is a **known** parametric form with substantively meaningful parameters — Michaelis-Menten, four-parameter sigmoid, exponential decay, Hill equation, growth curves.

## Levenberg-Marquardt (Marquardt 1963)

Damped Gauss-Newton that interpolates between gradient descent (far from optimum) and Gauss-Newton (near optimum):

```
(JᵀJ + λ I) δ = Jᵀ r
```

- `λ` **large** → gradient-descent behavior (safe far from the optimum).
- `λ` **small** → Gauss-Newton behavior (fast near the optimum).

Standard schedule: shrink `λ` on successful steps, grow it on failures.

## Standard errors

Under Normal errors,

```
Cov(θ̂) ≈ σ̂² · (JᵀJ)⁻¹        σ̂² = RSS / (n − p)
```

where `J = ∂f/∂θ` at `θ̂`. Wald tests and t-intervals follow.

## Files

- `python/nonlinear_regression.py` — from-scratch Levenberg-Marquardt with numerical Jacobian and asymptotic SE. Demo: Michaelis-Menten recovers Vmax = 5.006 (true 5.0), Km = 2.05 (true 2.0), matching `scipy.optimize.curve_fit` exactly; four-parameter sigmoid recovers all four parameters within one SE.
- `r/nonlinear_regression.R` — base `nls()` (Gauss-Newton); robust variant `robustbase::nlrob()`.

## When to use

- **Substantive parametric models** — enzyme kinetics, pharmacokinetics, growth curves, saturation phenomena, dose-response.
- Whenever the model has **interpretable parameters** you want to estimate and report.
- **Interpolation / extrapolation** with a plausible mechanistic form.

## When NOT to use

- **No prior knowledge of `f`** — reach for a spline / GAM / kernel smoother instead.
- **Very small samples** — nonlinear standard errors are asymptotic; consider a profile-likelihood interval or bootstrap.
- **Multimodal target** — LM converges to a local optimum. Use multiple random starts.

## Assumptions & caveats

- **Starting values** matter — nonlinear LS has multiple local minima and can fail to converge from a bad start.
- **Normal errors** underlie the asymptotic SE. Robust alternatives exist (`nlrob`, `nlrq` for quantile).
- **Identifiability** — over-parameterized models (e.g. free Hill exponent + free EC50) can be non-identifiable; profile likelihood diagnoses.
- **Numerical Jacobian** is fine for small `p`; provide analytical Jacobian for speed and stability at large `p`.

## Run

```
python techniques/nonlinear-regression/python/nonlinear_regression.py
Rscript techniques/nonlinear-regression/r/nonlinear_regression.R
```

**Refs:** Marquardt, D.W. "An algorithm for least-squares estimation of nonlinear parameters." *SIAM J. Appl. Math.* 11(2), 431–441, 1963; Bates, D.M. & Watts, D.G. *Nonlinear Regression Analysis and Its Applications*, Wiley, 1988; Seber, G.A.F. & Wild, C.J. *Nonlinear Regression*, Wiley, 2003.

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
