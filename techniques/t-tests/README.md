# t-tests (Reference §3.4)

Compare a sample mean to a value, or two sample means to each other, when the data are approximately normal (or `n` is large enough for the CLT to kick in).

## Variants

| Test | H₀ | Statistic | df |
|------|----|-----------|----|
| **One-sample** | `mean(x) = μ₀` | `t = (x̄ − μ₀) / (s/√n)` | `n − 1` |
| **Two-sample, Student** (equal variances) | `mean(x₁) = mean(x₂)` | `t = (x̄₁ − x̄₂) / √(s_p² (1/n₁ + 1/n₂))` | `n₁ + n₂ − 2` |
| **Two-sample, Welch** (unequal variances) | `mean(x₁) = mean(x₂)` | `t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)` | Satterthwaite (formula below) |
| **Paired** | `mean(x₁ − x₂) = 0` | one-sample t on the differences | `n − 1` |

**Pooled variance** (Student's): `s_p² = ((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2)`.
**Welch–Satterthwaite df**: `df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)]`.

## Choosing the right variant

- Two independent groups, you suspect or know variances differ → **Welch** (and use it by default — it has correct Type I error when variances *are* equal too, with a tiny loss of power).
- Two independent groups, variances really are equal → Student is slightly more powerful.
- Same units measured twice (pre/post, twin pairs, matched cases) → **paired**.
- Comparing a sample mean to a known constant → one-sample.

## Assumptions

- Data approximately normal *or* `n` large enough for the CLT (rule of thumb: `n ≥ 30` per group; less if the data are already symmetric).
- Independent observations (within and between groups; paired test relaxes the *between* requirement).
- Continuous outcome.

If badly violated: see Wilcoxon / Mann–Whitney (Ch. 6, future batch), Yuen's trimmed t (`techniques/robust-location-scale`), or a bootstrap.

## Effect size

A significant test only says "the difference isn't zero" — report **Cohen's d** (or **Hedges' g** for small samples; `techniques/effect-sizes`) alongside the p-value and the CI on the mean difference.

## Files
- `python/t_tests.py` — from-scratch one-sample / Student / Welch / paired; compares against `scipy.stats.ttest_1samp` / `ttest_ind` / `ttest_rel`.
- `r/t_tests.R` — from-scratch + `stats::t.test` (its `var.equal=` and `paired=` arguments cover all four cases).
- `pyspark/t_tests.py` — two-sample test from group sufficient statistics (`groupBy → mean/var/count`), then closed-form t and Welch df on the driver. The standard "distributed sufficient statistics" pattern.

## Run
```
python techniques/t-tests/python/t_tests.py
Rscript techniques/t-tests/r/t_tests.R
python techniques/t-tests/pyspark/t_tests.py
```

**Refs:** Casella & Berger, *Statistical Inference*, 2nd ed., Cengage, 2002; Welch, "The Generalization of 'Student's' Problem When Several Different Population Variances Are Involved," *Biometrika* 34(1–2), 28–35, 1947.

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
