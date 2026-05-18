# One-Way ANOVA (Reference §3.8, §3.9)

Test whether the means of `k ≥ 2` groups are all equal. Three variants here, increasingly forgiving of unequal variances.

| Test | Assumes equal variances? | When |
|------|--------------------------|------|
| **Classic F** | Yes | All groups have similar SD; balanced or near-balanced design |
| **Welch's** | No | **Modern default** — robust to unequal variances; recommended unless you have strong reason to believe equal variances |
| **Brown–Forsythe F\*** | No | Alternative to Welch; uses a different df estimator |

## The math

Per-group statistics: `n_i`, `x̄_i`, `s_i²`. Grand mean: `x̄ = (Σ nᵢ x̄ᵢ) / N`.

**Classic**:
- `SS_between = Σ nᵢ (x̄ᵢ − x̄)²`
- `SS_within  = Σ (nᵢ − 1) sᵢ²`
- `F = (SS_b/(k−1)) / (SS_w/(N−k))` ~ `F(k−1, N−k)` under H₀.

**Welch**: weights `wᵢ = nᵢ / sᵢ²`, weighted grand mean `x̄_w = (Σ wᵢ x̄ᵢ) / Σwᵢ`. F = (weighted SS between) / (variance-correction term), with a Satterthwaite-style `df₂`.

**Brown–Forsythe**: same numerator as classic; denominator is `Σ (1 − nᵢ/N) sᵢ²`, with Satterthwaite `df₂`.

## Effect size

Always report alongside the p-value (see `techniques/effect-sizes` for full details and benchmarks):
- `η² = SS_between / SS_total` — upward-biased
- `ω² = (SS_b − (k−1)·MS_w) / (SS_total + MS_w)` — less biased
- Cohen's `f = √(η²/(1−η²))`

## What comes next

A significant overall ANOVA only says "at least one pair of means differs." To find **which** pairs, use a post-hoc procedure:
- equal variances → **Tukey HSD** (`techniques/post-hoc-tests`)
- unequal variances → **Games–Howell** (`techniques/post-hoc-tests`)
- treatments vs. a single control → **Dunnett** (`techniques/post-hoc-tests`)
- many pre-planned pairs → control the family-wise error or FDR (`techniques/multiple-comparisons`)

Check the assumptions before believing the result: **normality of residuals** (`techniques/normality-tests`) and **homogeneity of variance** (`techniques/homogeneity-of-variance`). If badly violated and groups can't be transformed: Kruskal–Wallis (Ch. 6, future batch) or a permutation test.

## Files
- `python/one_way_anova.py` — from-scratch classic, Welch, Brown–Forsythe (each with closed-form F and Satterthwaite df); compares against `scipy.stats.f_oneway`.
- `r/one_way_anova.R` — from-scratch versions + base `aov` (classic) and `oneway.test(var.equal = FALSE/TRUE)` (Welch / classic).
- `pyspark/one_way_anova.py` — distributed sufficient statistics (`groupBy → count/mean/var_samp`), closed-form F on the driver. Works for any `k`.

## Run
```
python techniques/one-way-anova/python/one_way_anova.py
Rscript techniques/one-way-anova/r/one_way_anova.R
python techniques/one-way-anova/pyspark/one_way_anova.py
```

**Refs:** Welch, "On the Comparison of Several Mean Values: An Alternative Approach," *Biometrika* 38(3/4), 330–336, 1951; Brown & Forsythe, "The Small Sample Behavior of Some Statistics Which Test the Equality of Several Means," *Technometrics* 16(1), 129–132, 1974.

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
