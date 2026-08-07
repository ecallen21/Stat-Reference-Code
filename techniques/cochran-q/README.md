# Cochran's Q Test (Reference §8.10)

Generalizes McNemar's test to **k ≥ 2 related samples of binary outcomes**. Each of `n` subjects is measured under `k` conditions → an `n × k` binary matrix.

```
H_0 : Pr(success) is the same under all k conditions
H_a : at least one condition differs
```

## Statistic (Cochran 1950)

```
Q = (k − 1) · (k Σ_j C_j² − (Σ_j C_j)²) / (k Σ_i R_i − Σ_i R_i²)
```

- `R_i` = row sum (successes for subject i, in [0, k])
- `C_j` = column sum (successes in condition j, in [0, n])

Under `H_0`, `Q ~ χ²(k − 1)` asymptotically. `k = 2` reduces to McNemar's chi-square.

## Post-hoc: pairwise McNemar

After a significant Q, run all `k(k − 1)/2` pairwise McNemar tests + Bonferroni or BH correction.

## Files

- `python/cochran_q.py` — from-scratch Q statistic + chi-square p-value + Bonferroni-adjusted pairwise McNemar. Demo (n = 30, k = 4, subject-correlated observations): Q = 16.10, p = 0.0011; matches `statsmodels.stats.contingency_tables.cochrans_q` exactly.
- `r/cochran_q.R` — `nonpar::cochran.q` or a hand-rolled recipe using base R.

## When to use

- **Repeated-measures binary** outcome: same subjects tested under multiple conditions / timepoints / raters.
- **Cross-over trials** with binary primary endpoint.
- **Screening tests / diagnostic panels** — k tests applied to n subjects, checking whether positivity rates differ.

## Assumptions & caveats

- **Binary outcomes** only. For ordinal / continuous repeated measures use Friedman (already implemented) or a mixed model.
- **Same subjects across all k conditions** (no missing cells). Handle missingness with a GLMM (`lme4::glmer(y ~ condition + (1 | subject), family = binomial)`).
- **Chi-square approximation** works well when `n R̄ (k − R̄) ≥ ~24`; use exact permutation for small samples.

## Contrast

- **McNemar** — two related samples (Cochran's Q for k = 2 reduces to McNemar's χ²).
- **Friedman** — k related samples with ordinal / continuous outcomes.
- **GEE / mixed-effects binomial** — allows covariates and more flexible modeling of the correlation structure.

## Run

```
python techniques/cochran-q/python/cochran_q.py
Rscript techniques/cochran-q/r/cochran_q.R
```

**Refs:** Cochran, W.G. "The comparison of percentages in matched samples." *Biometrika* 37(3/4), 256–266, 1950; Fleiss, J.L. *Statistical Methods for Rates and Proportions*, 2nd ed., Wiley, 1981.

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
