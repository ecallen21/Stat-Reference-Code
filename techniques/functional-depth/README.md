# Functional Depth + Outlier Detection (Reference §31.9)

**Rank curves by centrality** and use the depth-boxplot to flag
outliers. López-Pintado & Romo (2009) introduced the **modified band
depth (MBD)**; Fraiman-Muñiz (2001) is an integrated pointwise-depth
alternative.

## Modified band depth (order 2)

For every pair `(x_a, x_b)`, the *band* at `t` is
`[min(x_a(t), x_b(t)), max(x_a(t), x_b(t))]`. MBD of curve `x_i` is
the average fraction of `t` where `x_i` lies inside the band:

```
MBD(x_i) = C(n, 2)⁻¹  Σ_{a<b}  (1/T)  Σ_t  𝟙[ x_i(t) ∈ B(t; a, b) ].
```

The **functional boxplot** shades the central-50 % band (top-half by
depth) and flags curves beyond `median − 1.5 · IQR` of depth as
outliers.

## When to use

- **Curve datasets** — spectra, growth, gait, sensor waveforms.
- **Robust functional descriptive statistics** — depth median, depth
  quantiles.
- **Quality-control screening** — flag anomalous curves before
  downstream analysis.

## When NOT to use

- **Highly-warped curves** — align first (`curve-registration`);
  otherwise depth flags phase misalignment as an outlier.
- **Sparse / irregular curves** — need PACE-based depth.
- **Multivariate functional data** — extensions exist but need care.

## Files

- `python/functional_depth.py` — from-scratch order-2 MBD +
  functional-boxplot outlier flag. Demo: 29 normal `sin(2πt)` curves +
  1 outlier with 5× amplitude. **Outlier gets depth = 0.098 (rank
  30/30, lowest)**; boxplot flags 2 curves (including the true
  outlier).
- `r/functional_depth.R` — `fda.usc::depth.*`, `roahd::MBD` (R);
  `scikit-fda`, `fdasrsf` (Python).

## Assumptions & caveats

- **Order-2 vs higher-order MBD** — higher order = more nuanced but
  quadratically more expensive.
- **Boxplot threshold** — `1.5 · IQR` on depths is heuristic;
  adjust for your false-alarm tolerance.
- **Phase alignment** — required before depth if curves are warped.
- **Curse of dimension** — depth is well-defined in infinite dimensions
  but noisier as T grows.
- **Choice of depth** — Fraiman-Muñiz, hMode, integrated pointwise;
  MBD is the standard default.

## Related in this repo

- `functional-pca`, `functional-regression`, `functional-anova`,
  `functional-clustering`, `curve-registration` — FDA family (this
  batch).
- `outlier-tests`, `multivariate-outlier-detection` — multivariate
  cousins.
- `mm-estimators-robust`, `robust-regression` — robust-stats family.
- `ood-detection` — the ML sibling of outlier flagging.

## Run

```
python techniques/functional-depth/python/functional_depth.py
Rscript techniques/functional-depth/r/functional_depth.R
```

**Refs:** López-Pintado, S. & Romo, J. "On the concept of depth for functional data." *JASA*, 2009; Fraiman, R. & Muñiz, G. "Trimmed means for functional data." *Test*, 2001; Sun, Y. & Genton, M.G. "Functional boxplots." *Journal of Computational and Graphical Statistics*, 2011.

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
