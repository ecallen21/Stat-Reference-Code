# Slice Sampler (Reference §25.11)

Neal (2003). A tuning-free MCMC method. At each step, sample a
**horizontal level** `u ~ U(0, f(x))`, then a new `x` uniformly from
the **slice** `S = {x' : f(x') > u}`.

Guarantees detailed balance with no accept/reject step.

## 1-D stepping-out + shrinkage

1. Draw `log u = log f(x) − Exp(1)`.
2. Grow an interval `[L, R]` around `x` until both endpoints fall
   below `log u`.
3. Repeatedly sample `x' ~ U(L, R)`; if `log f(x') > log u`,
   accept, else shrink `[L, R]` to exclude `x'`.

## Files

- `python/slice_sampler.py` — 1-D slice sampler (stepping-out +
  shrinkage) from scratch. Demo (bimodal target 0.4 N(−2, 1) + 0.6
  N(3, 0.5)): sample mean 0.85 vs analytic 1.00; sample var 6.68 vs
  6.55; P(X < 0) 0.42 vs analytic 0.39; lag-1 autocorr 0.87 with no
  tuning.
- `r/slice_sampler.R` — `mcmc::slice.sample`, `nimble`, MCMCglmm
  (R); `pymc.step_methods.Slice`, numpyro slice (Python).

## When to use

- **Univariate conditional posteriors** — inside Gibbs sampling
  (Neal 2003 originally motivated).
- **Non-standard 1-D densities** — bimodal, skewed, bounded.
- **Tuning aversion** — no step size to hand-set.

## When NOT to use

- **Very high dimensions** — component-wise slice is slow; HMC/NUTS
  is more efficient.
- **Densities cheap to evaluate but with sharp cliffs** — slice
  needs many evaluations per step near boundaries.

## Assumptions & caveats

- **Log-density evaluable up to constant** — same as MH.
- **Stepping-out width `w`** — tune it once to typical slice width;
  robustness ensured by shrinkage.
- **Multivariate extension** — apply 1-D slice to each coordinate
  in turn (Gibbs style) or use elliptical-slice for Gaussian
  priors.
- **Boundary handling** — reflected or transformed variables can
  keep `x` in support.

## Related in this repo

- `mcmc-metropolis-hastings`, `gibbs-sampler`, `hmc-nuts`,
  `adaptive-metropolis-haario` — MCMC family.
- `variational-inference`, `nested-sampling` — alternative Bayesian
  computation.
- `importance-sampling`, `bridge-sampling-evidence` — normalising-
  constant siblings.

## Run

```
python techniques/slice-sampler/python/slice_sampler.py
Rscript techniques/slice-sampler/r/slice_sampler.R
```

**Refs:** Neal, R.M. "Slice sampling." *Annals of Statistics*, 31(3): 705-767, 2003; Damlen, P.J., Wakefield, J.C. & Walker, S.G. "Gibbs sampling for Bayesian non-conjugate and hierarchical models by using auxiliary variables." *JRSS-B*, 61(2): 331-344, 1999.

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
