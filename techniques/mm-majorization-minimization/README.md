# MM — Majorization-Minimization (Reference §47.322)

Hunter & Lange (2004); Lange (2016). Instead of minimising
`f(x)` directly, iteratively minimise a SURROGATE `g(x | x_k)`
that MAJORIZES `f`: `g ≥ f` and `g(x_k | x_k) = f(x_k)`. Then

```
x_{k+1} = argmin_x g(x | x_k)
```

guarantees `f(x_{k+1}) ≤ f(x_k)`. EM, IRLS, and ISTA are all
MM instances. Choosing a good surrogate makes each step
easier than the original.

## Files

- `python/mm_majorization_minimization.py` — Two demos: (1)
  L1-median via Weiszfeld-style MM on 50 samples with 2
  outliers — MM = 0.14 (robust) vs sample mean = −0.05 pulled
  by outliers. (2) LASSO via MM on n=200, d=30 with 5 true
  nonzeros — objective drops smoothly to 0.578 in ~50
  iterations.
- `r/mm_majorization_minimization.R` — `glmnet` / `ncvreg`
  (MM/coord descent), Rcpp custom, `lme4` IRLS (R); sklearn /
  statsmodels IRLS, `cvxpy`, from-scratch (Python).

## When to use

- **Non-smooth / hard-to-differentiate objectives** with a
  natural surrogate.
- **Robust regression** — Huber-loss majorised by weighted
  least squares.
- **Mixture models** — EM is the classical MM.
- **Penalised regression** — LASSO / SCAD via quadratic
  majorisation.

## When NOT to use

- **When gradient / Newton method works** — MM adds a
  surrogate layer that may slow convergence.
- **Non-descent surrogates** — MM guarantees monotone decrease
  only if the surrogate genuinely majorises.
- **Very high-dimensional problems** — surrogate solve may
  itself be expensive.

## Assumptions & caveats

- **Tangent condition** `g(x_k | x_k) = f(x_k)`.
- **Majorisation** `g(x | x_k) ≥ f(x)` — proof-of-descent
  step.
- **Choice of surrogate** matters — quadratic surrogates give
  weighted-least-squares subproblems.
- **Local vs global** — MM produces a stationary point; global
  minimum only under convexity.

## Related in this repo

- `em-algorithm-mixture` — MM applied to mixture likelihoods.
- `proximal-gradient-method`, `fista-accelerated-proximal` —
  MM instances for composite convex problems.
- `robust-regression` — MM used to solve M-estimation.
- `admm-consensus` — related dual-decomposition alternative.

## Run

```
python techniques/mm-majorization-minimization/python/mm_majorization_minimization.py
Rscript techniques/mm-majorization-minimization/r/mm_majorization_minimization.R
```

**Refs:** Hunter, D.R. and Lange, K. "A tutorial on MM algorithms." *Am. Stat.*, 58(1): 30-37, 2004; Lange, K. *MM Optimization Algorithms*, SIAM, 2016.

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
