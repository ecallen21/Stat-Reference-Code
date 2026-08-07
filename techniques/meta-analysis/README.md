# Meta-Analysis (Reference §20.1)

Combine effect-size estimates `y_i` from `k` studies with within-study variances `v_i` to produce a pooled estimate and heterogeneity diagnostics.

## Fixed-effect (inverse-variance)

Assumes **all studies estimate the same true effect**:

```
w_i    = 1 / v_i
ȳ      = Σ w_i y_i / Σ w_i
Var(ȳ) = 1 / Σ w_i
```

## Random-effects (DerSimonian-Laird 1986)

Studies estimate **exchangeable** true effects `θ_i ~ N(μ, τ²)`. Method-of-moments τ²:

```
Q      = Σ w_i (y_i − ȳ_FE)²
τ̂²    = max(0, (Q − (k − 1)) / (Σ w_i − Σ w_i² / Σ w_i))
w_i^*  = 1 / (v_i + τ̂²)
μ̂      = Σ w_i^* y_i / Σ w_i^*
```

## Heterogeneity

```
Q       ~ χ²(k − 1) under H_0 : τ² = 0
I²      = max(0, (Q − (k − 1)) / Q) · 100        % variance from between-study
```

Interpretation: I² ≈ 25% low, 50% moderate, 75% high.

## Files

- `python/meta_analysis.py` — fixed-effect + DerSimonian-Laird random-effects + Q statistic + I². Demo (k = 8 studies, true μ = 0.35, τ = 0.15): pooled ≈ 0.49; homogeneity not rejected in this small-k demo.
- `r/meta_analysis.R` — `metafor::rma(yi, vi, method = "DL")` for the canonical R implementation with forest plots and PET-PEESE.

## Modern extensions

- **REML / ML** estimation of τ² (usually preferred over DL for real data).
- **Hartung-Knapp-Sidik-Jonkman (HKSJ)** adjustment — better SEs when `k` is small.
- **Bayesian meta-analysis** — full posterior for μ and τ².
- **Meta-regression** — model between-study heterogeneity as a function of study-level covariates.
- **Multivariate / network meta-analysis** — multiple outcomes or indirect comparisons.

## Diagnostics

- **Forest plot** — study effects + CI + pooled effect.
- **Funnel plot** — visual for publication bias.
- **Egger's test** — regression-based test for funnel-plot asymmetry.
- **Trim-and-fill / PET-PEESE** — publication-bias corrections.

## Assumptions & caveats

- **Comparable effect sizes** across studies — same outcome, similar scale.
- **Independent studies** — nested / overlapping cohorts violate this.
- **Publication bias** cannot be ruled out without an unbiased search strategy.
- **Reporting** should include: study-level table, forest plot, heterogeneity stats, funnel plot / publication-bias check, sensitivity analysis (leave-one-out).

## Run

```
python techniques/meta-analysis/python/meta_analysis.py
Rscript techniques/meta-analysis/r/meta_analysis.R
```

**Refs:** DerSimonian, R. & Laird, N. "Meta-analysis in clinical trials." *Control. Clin. Trials* 7(3), 177–188, 1986; Higgins, J.P.T. & Thompson, S.G. "Quantifying heterogeneity in a meta-analysis." *Stat. Med.* 21(11), 1539–1558, 2002; Viechtbauer, W. "Conducting meta-analyses in R with the metafor package." *J. Stat. Softw.* 36(3), 1–48, 2010.

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
