# Cramer's V and the Phi Coefficient (Reference §4.10)

Effect-size measures for association in a **categorical** contingency table — the natural companion to the chi-square test of independence (`techniques/chi-square-tests`).

## Definitions

For an `r × c` table of counts:

| Quantity | Formula | Range |
|----------|---------|-------|
| **Phi** φ (2×2 only) | `√(χ²/n)`; signed form `(ad − bc) / √((a+b)(c+d)(a+c)(b+d))` | `[−1, 1]` |
| **Cramer's V** | `√(χ² / (n · min(r−1, c−1)))` | `[0, 1]` |

For 2×2 tables, `|φ| = V`. For larger tables only V is meaningful; φ as a single number doesn't generalize.

## Bias correction (Bergsma 2013)

V is upward-biased in small samples — a non-significant chi-square can still yield a non-tiny V. The corrected version:
`V_corr = √(max(0, φ² − (r−1)(c−1)/(n−1)) / min(r̃ − 1, c̃ − 1))`
with `r̃ = r − (r−1)²/(n−1)` and `c̃ = c − (c−1)²/(n−1)`. Always report the corrected value with small `n`.

## Conventional benchmarks (small / medium / large)

| df = min(r−1, c−1) | small | medium | large |
|---------------------|-------|--------|-------|
| 1 | 0.10 | 0.30 | 0.50 |
| 2 | 0.07 | 0.21 | 0.35 |
| 3 | 0.06 | 0.17 | 0.29 |

These are Cohen's benchmarks — context-dependent as always.

## Relation to other techniques

- **Chi-square test** (`techniques/chi-square-tests`) — V is essentially χ² rescaled. Report the test for *whether* there's association, V for *how strong*.
- **Polychoric / tetrachoric** (`techniques/polychoric-correlation`) — if the categorical variables are really thresholded continuous variables (e.g. Likert), the polychoric correlation captures the underlying association more faithfully than V.
- **Cramer's V vs. Pearson r for binary variables**: `|φ| = |r|` for `0/1`-coded variables.

## Files
- `python/cramers_v_phi.py` — from-scratch χ², φ (signed) for 2×2, V, and Bergsma's bias-corrected V; compares against `scipy.stats.contingency.association`.
- `r/cramers_v_phi.R` — from-scratch + `DescTools::CramerV` (with `method = "ncchisqadj"` for the bias correction) and `vcd::assocstats`.
- PySpark: N/A — `χ²` from the contingency table aggregation in `techniques/chi-square-tests/pyspark/`; V is a one-line transform of `χ²` and `n`.

## Run
```
python techniques/cramers-v-phi/python/cramers_v_phi.py
Rscript techniques/cramers-v-phi/r/cramers_v_phi.R
```

**Refs:** Cramer, *Mathematical Methods of Statistics*, Princeton UP, 1946; Bergsma, "A Bias Correction for Cramer's V and Tschuprow's T," *J. Korean Statistical Soc.* 42(3), 323–328, 2013.

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
