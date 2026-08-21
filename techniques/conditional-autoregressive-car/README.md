# Conditional Autoregressive (CAR) Model (Reference §23.10)

Alternative to SAR for spatial random-effects modelling. Specifies **conditional distributions** rather than the joint:

```
u_i | u_{−i} ~ N( Σ_{j ~ i} b_ij · u_j / m_i,   τ² / m_i )
```

where `m_i` is the number of neighbours of location `i`. Common choice: `b_ij = 1`.

## Joint distribution

```
u ~ N(0, τ² · (D − α W)⁻¹)
```

- `D` = diagonal number-of-neighbours matrix.
- `α ∈ (0, 1)` controls autocorrelation; `α → 1` gives the **improper ICAR** used in BYM disease mapping.

## Besag-York-Mollié (BYM) disease mapping

```
y_i     ~ Poisson(E_i · exp(β + u_i + v_i))
u_i     : spatially-structured CAR
v_i     : unstructured Normal
```

Requires MCMC (`CARBayes`, `spBayes`) or **INLA** (fast Laplace approximation for latent Gaussian models — see `laplace-approximation`).

## Files

- `python/conditional_autoregressive_car.py` — CAR precision-matrix constructor + field simulation + ICAR quadratic penalty. Demo on 8×8 grid (rook adjacency, α = 0.95): drawn field has cor(u_i, avg neighbour) = 0.68; ICAR penalty = 51.
- `r/conditional_autoregressive_car.R` — `CARBayes::S.CARleroux`, `INLA` BYM, `spdep::spautolm(family='CAR')`.

## CAR vs SAR

- **CAR** specifies conditional distributions → natural for Gibbs sampling and hierarchical Bayes.
- **SAR** specifies the joint → natural for maximum-likelihood spatial econometrics.
- Both give proper Gaussian joint distributions; parameterizations differ.
- BYM is the standard **disease-mapping** application of CAR.

## Assumptions & caveats

- **Graph structure fixed** — chosen adjacency; sensitivity checks matter.
- **Propriety**: `α < 1` gives a proper joint; ICAR (`α = 1`) is improper — only valid as a prior with a sum-to-zero constraint.
- **Bayesian inference typically needed** — MLE for CAR exists (`spdep::spautolm`) but Bayesian is standard.

## Run

```
python techniques/conditional-autoregressive-car/python/conditional_autoregressive_car.py
Rscript techniques/conditional-autoregressive-car/r/conditional_autoregressive_car.R
```

**Refs:** Besag, J. "Spatial interaction and the statistical analysis of lattice systems." *J. R. Stat. Soc. B* 36(2), 192–236, 1974; Besag, J., York, J. & Mollié, A. "Bayesian image restoration, with two applications in spatial statistics." *Ann. Inst. Stat. Math.* 43(1), 1–20, 1991.

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
