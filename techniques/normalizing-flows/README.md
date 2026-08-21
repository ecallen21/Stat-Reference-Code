# Normalising Flows (Reference §27.x extra)

Express an intractable density `p_X(x)` as a **learned bijection** of a
tractable base density `p_Z(z)` (usually `N(0, I)`):

```
x = f_θ(z),   z = f_θ⁻¹(x)
log p_X(x) = log p_Z(z) − log |det J_{f}(z)|
```

The two workhorses:

- **Sampling** — draw `z ~ p_Z`, push forward through `f_θ`.
- **Density evaluation / likelihood training** — push `x` back through `f_θ⁻¹`, plug into the change-of-variables identity.

## RealNVP coupling layer (Dinh 2016)

Split `x = (x_a, x_b)`:

```
y_a = x_a
y_b = x_b ⊙ exp(s(x_a)) + t(x_a)
log|det J| = Σ_i s(x_a)_i
```

- Invertible by construction: `x_b = (y_b − t(y_a)) · exp(−s(y_a))`.
- Jacobian is triangular; log-det is a cheap `sum`.
- Alternate masks across layers (`[1,0]`, `[0,1]`, `[1,0]`, …) so every coordinate is transformed.
- `s(⋅), t(⋅)` are arbitrary neural nets — expressive without needing to be invertible themselves.

## Flow families

| Family | Fast sampling | Fast density |
|---|---|---|
| **RealNVP / Glow** | ✅ | ✅ |
| **MAF** (Papamakarios 2017) | ❌ | ✅ |
| **IAF** (Kingma 2016) | ✅ | ❌ |
| **Neural Spline Flow** (Durkan 2019) | ✅ | ✅ (higher-capacity than RealNVP) |
| **Continuous / Neural ODE Flow** (Chen 2018) | slow | slow (ODE solve) |

## When to use

- **Exact-likelihood generative modelling** — anywhere you need `log p(x)`, not just samples.
- **Anomaly / OOD detection** — low `log p(x)` under a flow trained on normal data.
- **Density estimation** on tabular data.
- **Prior for VAE / RL** — flow priors add expressiveness at the cost of KL tractability.
- **Simulation-based inference** — SNPE / SNL family use flows as approximate posteriors.
- **Modern generative alternative**: diffusion beats flows on image samples; flows remain preferred when you need exact likelihood.

## Files

- `python/normalizing_flows.py` — from-scratch 6-layer RealNVP with alternating masks. Because numerical-gradient training is fragile, the demo verifies the **RealNVP invariants** on a randomly-initialised flow:
  - Round-trip: `‖x − f(f⁻¹(x))‖ = 4e-16` (machine precision).
  - Log-det symmetry: `|log|det J_fwd| + log|det J_inv|| = 5e-17`.
  - Change-of-variables identity: `|direct log p(x) − base log p(z) + log|det J||= 3e-15`.
- `r/normalizing_flows.R` — `torch` (manual bijections); Python `nflows`, `FrEIA`, `tensorflow-probability` bijectors, `pyro.distributions.transforms`.

## Assumptions & caveats

- **Same-dim base and target** — flows preserve dimension. Use padding / dequantisation for discrete data.
- **Coupling layers move less mass per layer than affine transforms would** — need many stacked layers for high-capacity densities.
- **Log-det numerical stability** — tanh-bounded `s(⋅)` prevents runaway scales; clamping is standard.
- **Training** in practice is easy with autograd — the demo skips it for portability but the same code plugs into torch or JAX.
- **Not the state-of-the-art for image samples** — diffusion / EDM / flow-matching now beat flows on FID; flows win when exact likelihood matters.
- **Flow matching / rectified flow** (Lipman 2023) is the modern continuous-time cousin — often trained more easily than diffusion.

## Related in this repo

- `variational-autoencoder`, `gan-training`, `diffusion-model` — sibling generative families.
- `autoencoder` — deterministic non-probabilistic cousin.
- `laplace-approximation`, `variational-inference` — approximate-posterior alternatives.
- `residual-connections` — RealNVP layers are residual + coupling.

## Run

```
python techniques/normalizing-flows/python/normalizing_flows.py
Rscript techniques/normalizing-flows/r/normalizing_flows.R
```

**Refs:** Dinh, L., Sohl-Dickstein, J. & Bengio, S. "Density estimation using RealNVP." *ICLR*, 2017; Papamakarios, G., Pavlakou, T. & Murray, I. "Masked autoregressive flow for density estimation." *NeurIPS*, 2017; Durkan, C. et al. "Neural spline flows." *NeurIPS*, 2019; Lipman, Y. et al. "Flow matching for generative modeling." *ICLR*, 2023.

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
