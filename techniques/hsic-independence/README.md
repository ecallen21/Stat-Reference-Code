# HSIC Kernel Independence Test (Reference §47.56)

Gretton, Bousquet, Smola & Schölkopf (2005). Hilbert-Schmidt
Independence Criterion — kernel dependence measure that detects
**any** nonlinear association:

    HSIC(X, Y) = ‖C_{XY}‖²_HS,   empirical form  = (n−1)⁻² tr(K H L H).

For characteristic kernels (e.g. Gaussian RBF) HSIC = 0 iff X ⟂ Y.
Gamma-approximation of the null gives an analytical p-value
(Gretton et al 2008); permutation test is exact.

## Files

- `python/hsic_independence.py` — RBF-kernel HSIC + gamma-
  approx p-value from scratch, median-heuristic bandwidth. Demo
  (n=300):
  - independent → HSIC 0.0004, p ≈ 1.00
  - linear     → HSIC 0.062, p ≈ 0
  - quadratic  → HSIC 0.043, p ≈ 0 (Pearson would miss)
  - sinusoid   → HSIC 0.006, p ≈ 0.0002 (still detected)
- `r/hsic_independence.R` — `dHSIC`, `energy::dcov` (R);
  `hyppo`, `pyRMT`, from-scratch (Python).

## When to use

- **Test independence without assuming linearity** — feature
  screening, MI-like measure.
- **Blind source separation** (ICA objective).
- **Causal discovery** — conditional-HSIC used in KCI-style
  independence oracles.
- **Kernel-based two-sample tests** (MMD sibling in same
  framework).

## When NOT to use

- **Very large n** — O(n²) memory / n³ compute for the full
  kernel; use random-Fourier features or Nyström for scale.
- **High-dim X or Y** — kernel bandwidth choice degrades;
  consider distance covariance instead.
- **Small n** — gamma approximation loose; use permutation
  p-value.

## Assumptions & caveats

- **Bandwidth** — median-heuristic is a common default; sensitivity
  matters when the association is highly localised.
- **Characteristic kernel** — Gaussian RBF is; polynomial is not
  (misses higher moments).
- **Bias correction** — empirical HSIC is biased; unbiased
  estimator exists (Song 2012).
- **Gamma vs permutation p** — gamma is asymptotic; use permutation
  for small n or extreme p-values.

## Related in this repo

- `mutual-information`, `conditional-mutual-info`,
  `transfer-entropy` — information-theoretic dependence measures.
- `kl-divergence`, `f-divergences` — distributional distances.
- `permutation-tests`, `distance-correlation` (see
  `pearson-correlation`) — nonparametric independence tests.
- `causal-discovery-pc`, `notears-dag-learning` — causal discovery
  uses HSIC as an oracle.

## Run

```
python techniques/hsic-independence/python/hsic_independence.py
Rscript techniques/hsic-independence/r/hsic_independence.R
```

**Refs:** Gretton, A., Bousquet, O., Smola, A. & Schölkopf, B. "Measuring statistical dependence with Hilbert-Schmidt norms." *ALT*, 2005; Gretton, A. et al. "A kernel statistical test of independence." *NeurIPS*, 2008.

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
