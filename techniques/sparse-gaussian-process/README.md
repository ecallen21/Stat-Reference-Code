# Sparse Gaussian Process (Inducing Points) (Reference §47.89)

Snelson & Ghahramani (2005, FITC); Titsias (2009, VFE/SVGP). Full
GPs scale O(n³). Replace with M ≪ n inducing pseudo-inputs Z; the
covariance is approximated by

    Q_nn = K_nm K_mm⁻¹ K_mn,

giving posterior mean / variance at O(n M² + M³). With M=50–500,
GPs handle n=10⁵+.

## Files

- `python/sparse_gaussian_process.py` — from-scratch FITC (fully
  independent training conditional) predictions on a 1-D `sin(x)`
  regression. Demo (n=500):
  - M=10: ‖μ_FITC − μ_full‖ / ‖μ_full‖ = 0.008
  - M=30: essentially exact (rel err 0.0000)
  - M=100: exact.
- `r/sparse_gaussian_process.R` — `mlegp`, `kernlab` for full GPs,
  no first-class SGP in R (R+reticulate wraps GPflow); Python via
  `GPflow`, `gpytorch`, from-scratch.

## When to use

- **GPs at n > a few thousand** — full O(n³) intractable.
- **Bayesian optimisation** where each iter refits with growing n.
- **Deep kernel learning** — SVGP scales with a neural feature
  extractor.
- **Multi-output GP / geostatistics** — inducing points shared
  across tasks.

## When NOT to use

- **Small n (< few hundred)** — full GP is cheaper and better.
- **Very rough functions** — inducing-point approximation smooths.
- **Non-Gaussian likelihoods** — need VFE + SVI (SVGP), FITC alone
  can be biased.
- **When you don't want to tune Z** — FITC has known pathologies at
  Z misplacement.

## Assumptions & caveats

- **Inducing-point placement** — grid, k-means, or learned jointly
  with hyperparameters; heavily affects fidelity.
- **FITC vs VFE** — FITC can be optimistic about variance; VFE (Titsias)
  is a lower bound on log marginal likelihood and pairs cleanly with SVI.
- **Hyperparameter learning** — optimise ell, sf², sn² jointly with Z.
- **Numerical stability** — jitter (~1e-6) on Kmm diagonal essential.

## Related in this repo

- `gaussian-process-regression`, `gaussian-process-latent-variable-model`
  — full-GP siblings.
- `bayesian-optimization`, `nadaraya-watson-kernel-regression`,
  `kernel-density-estimation`, `kernel-pca` — kernel-method
  neighbours.
- `random-fourier-features`, `random-projections` — alternative
  scalable kernel approximations.
- `variational-inference`, `hmc-nuts`, `laplace-approximation` —
  posterior-approximation toolkit.

## Run

```
python techniques/sparse-gaussian-process/python/sparse_gaussian_process.py
Rscript techniques/sparse-gaussian-process/r/sparse_gaussian_process.R
```

**Refs:** Snelson, E. & Ghahramani, Z. "Sparse Gaussian processes using pseudo-inputs." *NeurIPS*, 2005; Titsias, M.K. "Variational learning of inducing variables in sparse Gaussian processes." *AISTATS*, 2009; Hensman, J., Fusi, N. & Lawrence, N.D. "Gaussian processes for big data." *UAI*, 2013.

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
