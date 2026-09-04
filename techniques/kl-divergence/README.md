# Kullback-Leibler Divergence (Reference §34.3)

**KL(p ‖ q) = 𝔼_p[log(p/q)]** — expected excess log-likelihood cost of
using `q` when the truth is `p`. Foundational quantity of information
theory.

## Formulas

```
Discrete:    KL(p ‖ q) = Σ_x p(x) log(p(x) / q(x))
Continuous:  KL(p ‖ q) = ∫ p(x) log(p(x) / q(x)) dx
JS(p, q)     = ½ KL(p ‖ m) + ½ KL(q ‖ m),   m = ½(p + q)   ∈ [0, log 2]
```

Gaussian closed form: `KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)) = log(σ₂/σ₁) +
(σ₁² + (μ₁−μ₂)²)/(2σ₂²) − ½` (nats).

## Properties

- **Non-negative**, zero iff `p = q` (Gibbs inequality).
- **Asymmetric** — `KL(p ‖ q) ≠ KL(q ‖ p)`.
- **Not a metric** — no triangle inequality; JS is a symmetric proxy.
- **Zero-forcing** — infinite if `q(x) = 0` where `p(x) > 0`.

## When to use

- **MLE / variational inference** — the ELBO minimises `KL(q ‖ p_post)`.
- **Model calibration** — target `KL(p_data ‖ p_model)`.
- **Data drift** — see `data-drift-detection` (uses PSI, an
  approximation).
- **Regularisation** — KL to a prior in Bayesian nets, VAEs.

## Files

- `python/kl_divergence.py` — discrete KL, JS, Gaussian closed form,
  Monte-Carlo estimator. Demo:
  - `KL(p ‖ q) = 0.123` vs `KL(q ‖ p) = 0.133` (asymmetric).
  - JS = 0.032 (symmetric).
  - Gaussian analytic vs MC: **0.4431 vs 0.4449 nats**.
  - Zero-forcing case explodes to `18.93`.
- `r/kl_divergence.R` — `FNN`, `entropy`, `philentropy` (R);
  `scipy.special.rel_entr`, `torch.nn.functional.kl_div` (Python).

## Assumptions & caveats

- **Support match required** — smooth `q` to prevent infinities
  (Laplace smoothing).
- **Reverse KL is different** — VI often uses reverse KL, which is
  mode-seeking rather than mean-seeking.
- **Estimation from samples** — k-NN (Wang-Kulkarni-Verdu 2009) for
  continuous; large-sample bias-corrected estimators for discrete.
- **Rényi α-divergence** generalises KL; α = 1 recovers KL.

## Related in this repo

- `shannon-entropy` — `H(p) + KL(p ‖ q) = cross-entropy(p, q)`.
- `cross-entropy-log-loss` — the training-loss version.
- `f-divergences` — the family KL belongs to.
- `mutual-information` — `I(X; Y) = KL(p(X, Y) ‖ p(X) p(Y))`.
- `information-geometry` — KL is a squared "distance" locally.
- `variational-inference` (if present) — ELBO minimises KL.

## Run

```
python techniques/kl-divergence/python/kl_divergence.py
Rscript techniques/kl-divergence/r/kl_divergence.R
```

**Refs:** Kullback, S. & Leibler, R.A. "On information and sufficiency." *Annals of Mathematical Statistics*, 1951; Lin, J. "Divergence measures based on the Shannon entropy." *IEEE Transactions on Information Theory*, 1991.

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
