# Gaussian Graphical Model / Graphical LASSO (Reference §30.8)

For `X ~ N(0, Σ)`, the **precision matrix** `Ω = Σ⁻¹` has an
intuitive graph interpretation:

```
Ω_{ij} = 0   iff   X_i ⊥⊥ X_j  |  X_{-{i,j}}.
```

Zero entries in `Ω` mark **conditional-independence edges** in the
associated graph — the network psychometrics or gene-regulatory
network view.

## Graphical LASSO (Friedman-Hastie-Tibshirani 2008)

```
min_Ω  −log det Ω  +  tr(S Ω)  +  ρ · Σ_{i ≠ j} |Ω_ij|.
```

Coordinate-descent solves it in `O(d³)` per outer iteration; sparsity
pattern is the recovered graph.

## When to use

- **Network psychometrics** — dependency network of survey items
  (`qgraph`, `bootnet`).
- **Gene-regulatory / metabolic networks** — conditional independence
  as biological structure.
- **Portfolio construction** — sparse-precision covariance estimator.
- **Any high-dim `p ≥ n`** covariance estimation task.

## When NOT to use

- **Non-Gaussian data** — copula / nonparanormal extensions available
  (`huge::npn`).
- **Very correlated features** — GGM assumes conditional independence
  can be extracted; high multicollinearity breaks sparsity recovery.
- **Dynamic / time-varying dependencies** — see time-varying GGM
  (Hallac 2017).

## Files

- `python/gaussian_graphical_model.py` — from-scratch wrapper around
  `sklearn.covariance.GraphicalLasso`. Demo: synthetic precision
  matrix with two connected components (chain of 4 + pair of 2);
  `n = 400`, `d = 6`. **Edge recovery TP = 4, FP = 0, FN = 0** at
  `ρ = 0.05` — exact structure recovered.
- `r/gaussian_graphical_model.R` — `glasso`, `huge`, `qgraph`,
  `bootnet` (R); `sklearn`, `skggm` (Python).

## Assumptions & caveats

- **ρ selection** — BIC / eBIC / cross-validated log-likelihood
  (Foygel-Drton 2010).
- **Gaussian assumption** — check with q-q plots; use nonparanormal
  (Liu-Lafferty-Wasserman 2009) for skewed data.
- **Multiple comparisons on the edges** — for confirmatory inference,
  use debiased-GGM (Ren 2015) or `bootnet` stability.
- **Precision sign** — `Ω_ij < 0` = positive partial correlation;
  `Ω_ij > 0` = negative. Report signed partial correlations for
  interpretability.
- **Time-varying** — Hallac 2017 fused-GGM.

## Related in this repo

- `ridge-lasso-elasticnet`, `debiased-lasso`, `adaptive-lasso` — the
  L1 family GLASSO belongs to.
- `stochastic-block-model`, `latent-space-network`,
  `qap-network-regression`, `node2vec-deepwalk`, `patient-similarity-
  network` — network family (this batch).
- `covariance-estimation-*` — high-dim covariance sibling.
- `partial-least-squares`, `canonical-correlation` — related latent-
  structure methods.

## Run

```
python techniques/gaussian-graphical-model/python/gaussian_graphical_model.py
Rscript techniques/gaussian-graphical-model/r/gaussian_graphical_model.R
```

**Refs:** Friedman, J., Hastie, T. & Tibshirani, R. "Sparse inverse covariance estimation with the graphical LASSO." *Biostatistics*, 2008; Meinshausen, N. & Bühlmann, P. "High-dimensional graphs and variable selection with the LASSO." *Annals of Statistics*, 2006; Foygel, R. & Drton, M. "Extended Bayesian information criteria for Gaussian graphical models." *NeurIPS*, 2010.

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
