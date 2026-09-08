# Welch's t-Test (Reference §47.113)

Welch (1947). Two-sample t-test that does NOT assume equal
variances:

    t = (x̄ − ȳ) / √(s_x²/n_x + s_y²/n_y)
    df = (s_x²/n_x + s_y²/n_y)² / ((s_x²/n_x)² / (n_x−1) + (s_y²/n_y)² / (n_y−1))

(Satterthwaite df.) Preferred over Student's t in practice
(Delacre-Lakens-Leys 2017) even when variances look similar —
Welch is nearly as powerful under equal variance and drastically
safer under unequal variance + unbalanced n.

## Files

- `python/welchs_t_test.py` — from-scratch Welch's t + Student's
  pooled-variance baseline. Demo:
  - equal-variance normals: Welch p = 0.30 = Student p (matches).
  - unequal-var (1, 3): Welch p = 0.80 vs Student 0.80.
  - unequal-var + unbalanced n=(10, 200), σ=(1, 5): Welch p = 0.42
    vs Student p = 0.79.
  Type-I under H₀ mean-diff = 0 with σ = (1, 3), n = (20, 60):
  - **Welch's Type-I  ≈ 0.050 (target)**
  - Student's Type-I ≈ 0.003 (deflated -- CI too wide).
- `r/welchs_t_test.R` — `stats::t.test(var.equal=FALSE)` (R
  default);
  `scipy.stats.ttest_ind(equal_var=False)` (Python).

## When to use

- **DEFAULT two-sample mean comparison** — Delacre-Lakens-Leys
  recommend as the routine choice.
- **Balanced or unbalanced n**, equal or unequal variances.
- **Non-normal but symmetric-ish samples** — Welch is fairly
  robust; use permutation / bootstrap for heavy tails.

## When NOT to use

- **Paired data** — use paired t-test.
- **Highly skewed / heavy-tailed** — Mann-Whitney / bootstrap /
  Yuen's trimmed-mean t.
- **Multiple groups** — use ANOVA or Welch-ANOVA / Brown-Forsythe.

## Assumptions & caveats

- **Approximate normality** — Welch is fairly robust for n ≥ 30;
  bootstrap for smaller.
- **Satterthwaite df** — a real-valued approximation; ties well
  with exact distribution for reasonable n.
- **Effect size**: report Cohen's d / Hedges' g alongside p.
- **Not equivalent to Student's** — Welch loses ~1-2 % power when
  variances are equal + n balanced; a fair trade for robustness.

## Related in this repo

- `t-tests`, `equivalence-testing-tost`, `mann-whitney`,
  `wilcoxon-signed-rank`, `permutation-tests`, `sign-test`,
  `moods-median` — two-sample / rank tests.
- `levene-brown-forsythe`, `bland-altman-agreement`, `agreement-beyond-kappa`
  — variance / agreement diagnostics.
- `effect-sizes`, `mde-sample-size` — magnitude / power tooling.
- `one-way-anova`, `repeated-measures-anova`, `manova`,
  `ancova` — extensions to more groups / covariates.

## Run

```
python techniques/welchs-t-test/python/welchs_t_test.py
Rscript techniques/welchs-t-test/r/welchs_t_test.R
```

**Refs:** Welch, B.L. "The generalization of 'Student's' problem when several different population variances are involved." *Biometrika* 34(1-2): 28-35, 1947; Delacre, M., Lakens, D. & Leys, C. "Why psychologists should by default use Welch's t-test instead of Student's t-test." *International Review of Social Psychology* 30(1): 92-101, 2017.

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
