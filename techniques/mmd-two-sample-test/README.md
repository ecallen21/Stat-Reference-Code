# MMD Two-Sample Test (Reference §47.65)

Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012). Maximum
Mean Discrepancy in an RKHS:

    MMD²(P, Q) = ‖μ_P − μ_Q‖²_H
               = E[k(x,x′)] + E[k(y,y′)] − 2 E[k(x,y)].

For characteristic kernels (e.g. Gaussian RBF), MMD = 0 iff P = Q.
Detects any distributional shift — mean, variance, mixture — that
univariate tests can miss.

## Files

- `python/mmd_two_sample_test.py` — unbiased empirical MMD²
  (Gretton eq 3) + permutation p-value from scratch. Demo (m=n=200,
  d=2, 200 perms):
  - same N(0,1)         → MMD²=−0.003, p=0.82
  - mean shift +0.3     → MMD²=+0.052, p=0.005
  - variance shift 1.5× → MMD²=+0.037, p=0.005
  - mixture shift       → MMD²=+0.092, p=0.005.
- `r/mmd_two_sample_test.R` — `kernlab::kmmd`, `eummd` (R);
  `hyppo`, `torch-two-sample`, from-scratch (Python).

## When to use

- **Multivariate distribution comparison** — dataset drift,
  covariate shift, GAN evaluation.
- **Model-fit diagnostics** — MMD between generated and real
  samples.
- **Kernel-based causal-discovery oracles** — HSIC / KCI relatives.
- **Adversarial data detection**.

## When NOT to use

- **Very large n** — O(n²) memory; use random-Fourier features or
  linear-time MMD (block estimator).
- **Very high-dim data with tight budget** — bandwidth choice
  degrades; kernel-ME / classifier-two-sample-test alternatives.

## Assumptions & caveats

- **Characteristic kernel** — Gaussian RBF, Laplacian, Matérn are;
  polynomial is not.
- **Bandwidth** — median-heuristic default; sensitivity matters.
- **Unbiased estimator** can be negative (near null).
- **Permutation p-value** exact under H_0: P = Q; asymptotic
  gamma-approx also works.

## Related in this repo

- `hsic-independence` — kernel independence sibling.
- `kolmogorov-smirnov`, `permutation-tests`, `wilcoxon-signed-rank`,
  `mann-whitney` — univariate two-sample tests.
- `distance-correlation`, `mutual-information`, `f-divergences`,
  `kl-divergence` — dependence / divergence measures.
- `data-drift-detection`, `concept-drift-adwin`,
  `covariate-shift-adaptation` — dataset-drift toolkit.

## Run

```
python techniques/mmd-two-sample-test/python/mmd_two_sample_test.py
Rscript techniques/mmd-two-sample-test/r/mmd_two_sample_test.R
```

**Refs:** Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A. "A kernel two-sample test." *JMLR* 13: 723-773, 2012; Chwialkowski, K. et al. "Fast two-sample testing with analytic representations of probability measures." *NeurIPS*, 2015.

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
