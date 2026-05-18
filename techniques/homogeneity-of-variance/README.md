# Homogeneity of Variance (Reference §3.20, §3.55)

Check the "equal variances" assumption made by classic Student's t-test, classic one-way ANOVA, and Tukey HSD. Useful as a diagnostic — but for serious analyses, prefer the *variance-robust* version of the actual test (Welch's t, Welch's ANOVA, Games–Howell) over a pre-test → conditional decision pipeline.

## Tests

| Test | Statistic | Robust to non-normality? |
|------|-----------|---------------------------|
| **Levene** (center = mean) | One-way ANOVA on `\|xᵢⱼ − x̄ᵢ\|` | Moderate |
| **Brown–Forsythe** (center = median) | One-way ANOVA on `\|xᵢⱼ − median(xᵢ)\|` | **High — recommended default** |
| **Bartlett** | Likelihood-ratio χ² against equal variances | **Very low** — only valid if groups are clearly normal |
| **F-test of two variances** | `s₁²/s₂² ~ F(n₁−1, n₂−1)` | Very low (same caveat as Bartlett) |

## How to use them

1. Plot the groups (boxplot / strip chart) — a `5:1` ratio of largest to smallest SD is usually visible by eye.
2. If group sizes are roughly equal and the ratio is modest (< 3×), Student's t and classic ANOVA are remarkably forgiving — you can use them.
3. Otherwise, just **use Welch by default** (`techniques/t-tests`, `techniques/one-way-anova`) and skip the pre-test. Welch has correct type-I error when variances *are* equal, with only a tiny loss of power — there's no downside.
4. For post-hoc pairwise comparisons under unequal variances, use **Games–Howell** (`techniques/post-hoc-tests`).

## "Pre-testing" warning

Choosing between Student's and Welch's t based on a Levene/Bartlett result inflates the overall type-I error of the conditional procedure. The "always-Welch" strategy avoids this entirely. The tests here are for assumption *understanding*, not for gating a downstream test.

## Files
- `python/homogeneity_of_variance.py` — from-scratch Levene / Brown–Forsythe / Bartlett / two-sample F-test; compares against `scipy.stats.levene` and `scipy.stats.bartlett`.
- `r/homogeneity_of_variance.R` — from-scratch versions + `stats::bartlett.test`, `stats::var.test`, and `car::leveneTest` (which lets you pick the center).
- PySpark: N/A — for large samples, plot a Q-Q of the residuals from a stratified sub-sample; almost any formal HOV test will be significant on large `n`.

## Run
```
python techniques/homogeneity-of-variance/python/homogeneity_of_variance.py
Rscript techniques/homogeneity-of-variance/r/homogeneity_of_variance.R
```

**Refs:** Levene, "Robust Tests for Equality of Variances," in *Contributions to Probability and Statistics*, Stanford UP, 1960; Brown & Forsythe, "Robust Tests for the Equality of Variances," *JASA* 69, 364–367, 1974; Bartlett, "Properties of Sufficiency and Statistical Tests," *Proc. Royal Soc. A* 160, 268–282, 1937.

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
