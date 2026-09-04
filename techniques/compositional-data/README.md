# Compositional Data Analysis (Reference §38.2)

Aitchison (1986). Data that sum to a constant — proportions,
percentages, geochemical assays, microbiome counts — live on the
simplex. Classical statistics on raw parts are misleading (spurious
correlation, wrong distances). Aitchison's remedy: transform to real
coordinates via log-ratios.

## Transformations

Given `x ∈ Simplex^D`:

- **ALR** — `alr(x)_i = log(x_i / x_D)`. Non-symmetric; drops one
  part.
- **CLR** — `clr(x)_i = log(x_i / g(x))` with `g` the geometric mean.
  Symmetric; rows sum to 0 (rank-deficient).
- **ILR** — orthonormal basis via a sequential binary partition;
  full-rank and isometric with respect to Aitchison geometry
  (Egozcue et al. 2003).

## Aitchison distance

```
d_A(x, y) = || clr(x) − clr(y) ||_2
```

Invariant to perturbation and scale; the correct metric on the
simplex.

## When to use

- **Microbiome / genomics** (OTU / ASV relative abundances).
- **Geochemistry, mineralogy, sediment composition**.
- **Household budget shares, election vote shares, market shares**.
- **Any data measured as proportions** whose parts are inherently
  linked by a sum-to-constant constraint.

## When NOT to use

- **Data are truly independent counts / totals** — model on the
  original scale (Poisson / NB).
- **Structural zeros** where a part is impossible — no log-ratio
  fix; use zero-inflated or partitioned models.

## Files

- `python/compositional_data.py` — closure, ALR, CLR, ILR,
  Aitchison distance. Demo: 6 Dirichlet compositions; shows raw-
  correlation vs CLR-correlation and Aitchison vs Euclidean distance
  (Aitchison distance 2.30 vs Euclidean 0.39).
- `r/compositional_data.R` — `compositions`, `robCompositions`,
  `zCompositions` (R); `scikit-bio` (Python).

## Assumptions & caveats

- **Rounded zeros** in count-derived compositions need zero
  replacement (`zCompositions::cmultRepl`) before log-ratios.
- **Sub-composition invariance** — CLR is basis-dependent (changes
  when parts are added/removed); prefer ILR for invariant analyses.
- **Interpretation** — coefficients live in log-ratio space; report
  as ratios or use back-transform to display on the simplex.
- **Small `D`** — ILR partition choice matters; PBBS / balances
  should be scientifically motivated.

## Related in this repo

- `dirichlet-regression` (if present) — parametric alternative.
- `pca`, `k-means` — perform on ILR coordinates for compositional
  data.
- `beta-regression` — single-proportion analogue.

## Run

```
python techniques/compositional-data/python/compositional_data.py
Rscript techniques/compositional-data/r/compositional_data.R
```

**Refs:** Aitchison, J. *The Statistical Analysis of Compositional Data*, Chapman & Hall, 1986; Pawlowsky-Glahn, V., Egozcue, J.J., & Tolosana-Delgado, R. *Modeling and Analysis of Compositional Data*, Wiley, 2015; Egozcue, J.J. et al. "Isometric logratio transformations for compositional data analysis." *Mathematical Geology*, 2003.

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
