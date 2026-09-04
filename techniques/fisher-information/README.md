# Fisher Information + Cramér-Rao Bound (Reference §34.4)

For a parametric family `p(x; θ)` with score
`s(x; θ) = ∂/∂θ log p(x; θ)`:

```
I(θ)  =  𝔼[ s(x; θ) s(x; θ)ᵀ ]  =  −𝔼[ ∂² log p / ∂θ ∂θᵀ ].
```

## Cramér-Rao lower bound

Any **unbiased** estimator `θ̂` satisfies

```
Var(θ̂)  ≥  I(θ)⁻¹ / n
```

MLE attains this bound asymptotically (for regular families).

## When to use

- **Asymptotic SEs** for MLE — standard practice in every statistical
  package via the inverse observed information.
- **Optimal experimental design** — maximise `det I(θ)` (D-optimality).
- **Information-geometric optimisation** (natural gradient) — see
  `information-geometry`.
- **Bounds for benchmark estimators**.

## When NOT to use

- **Biased estimators** (shrinkage / regularised) can beat the CRB in
  MSE.
- **Non-regular families** (boundary parameters, unidentified
  parameters) — CRB may not apply.
- **Small n** — the asymptotic bound may be loose.

## Files

- `python/fisher_information.py` — analytic Fisher info for Gaussian
  `(μ, σ²)`; empirical MLE variance vs CRB over `B = 5000` sims,
  `n = 50`, `μ = 1.5`, `σ² = 2.0`:
  - **Var(μ̂) empirical 0.0408 vs CRB 0.0400** — matches.
  - **Var(σ̂²) empirical 0.1487 vs CRB 0.1600** — close (small-n
    negative bias).
  - Shrinkage MSE demo: pulling μ̂ 10 % toward 0 can beat MLE
    variance at some bias cost.
- `r/fisher_information.R` — `numDeriv::hessian`, `stats::vcov` (R);
  `statsmodels.tools.numdiff.approx_hess`, `torch.autograd` (Python).

## Assumptions & caveats

- **Regularity conditions** — support of `p` doesn't depend on `θ`;
  differentiability under the integral.
- **Observed vs expected info** — `sample_hessian ≈ I(θ)` for large n.
- **Multi-parameter CRB** — matrix inequality `Var ≥ I⁻¹`.
- **Confidence intervals** — asymptotic Wald `θ̂ ± z √(I⁻¹/n)`;
  profile / LR CIs for small n.

## Related in this repo

- `shannon-entropy`, `kl-divergence`, `mutual-information` — sister
  info-theoretic quantities.
- `information-criteria` — AIC = − 2 log L + 2 p uses the same
  ideas.
- `information-geometry` — Fisher info as Riemannian metric.
- `delta-method`, `sandwich-robust-se` — SE-related methods.
- `likelihood-ratio-tests` — Wilks' theorem uses Fisher info.

## Run

```
python techniques/fisher-information/python/fisher_information.py
Rscript techniques/fisher-information/r/fisher_information.R
```

**Refs:** Fisher, R.A. "On the mathematical foundations of theoretical statistics." *Philosophical Transactions of the Royal Society A*, 1922; Cramér, H. *Mathematical Methods of Statistics*, Princeton University Press, 1946; Rao, C.R. "Information and the accuracy attainable in the estimation of statistical parameters." *Bull Cal Math Soc*, 1945.

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
