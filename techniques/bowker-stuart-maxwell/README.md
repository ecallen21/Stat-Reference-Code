# Bowker's Symmetry + Stuart–Maxwell Marginal Homogeneity (Reference §8.7, §8.15)

Both extend **McNemar's test** from 2×2 to K×K *paired* tables (same categories on rows and columns: before/after, rater1/rater2, ...).

## The two null hypotheses

For a K×K paired table with cell counts `n_ij`:

- **Bowker (symmetry)**: `H₀: p_ij = p_ji for all i ≠ j`. Every off-diagonal cell equals its mirror. The strong hypothesis.
- **Stuart–Maxwell (marginal homogeneity)**: `H₀: row_i = col_i for all i`. Each category has the same overall probability on rows and columns. The weaker hypothesis.

Symmetry implies marginal homogeneity but not vice versa — so a significant Bowker + non-significant Stuart–Maxwell means "individual pairs are asymmetric, but the marginals happen to balance out."

## Formulas

**Bowker**:
```
X²  =  Σ_{i<j}  (n_ij − n_ji)² / (n_ij + n_ji)      ~ χ²_{K(K−1)/2}
```
Reduces to McNemar's χ² (no CC) when K = 2. Empty pairs (`n_ij + n_ji = 0`) contribute 0 by convention; df unchanged.

**Stuart–Maxwell**:
```
d_i = row_i − col_i        (drop the last; the K values sum to 0)
V_ii = row_i + col_i − 2·n_ii
V_ij = −(n_ij + n_ji)      for i ≠ j
X²   = d' V⁻¹ d            ~ χ²_{K−1}
```

## Files

- `python/bowker_stuart_maxwell.py` — both tests from scratch. Bowker matches `statsmodels.stats.contingency_tables.SquareTable(shift_zeros=False).symmetry()` to 12 decimals. Stuart–Maxwell matches statsmodels with `shift_zeros=False` (statsmodels' default is to add 0.5 to zero cells; the file cross-checks both variants).
- `r/bowker_stuart_maxwell.R` — from-scratch + `stats::mcnemar.test` (which computes Bowker's statistic on K×K tables).

## Assumptions

- Same categorization scheme on rows and columns (they must be the same K categories).
- Independence across pairs (rows of raw data), same as McNemar.

## Run

```
python techniques/bowker-stuart-maxwell/python/bowker_stuart_maxwell.py
Rscript techniques/bowker-stuart-maxwell/r/bowker_stuart_maxwell.R
```

**Refs:** Bowker, A.H. "A test for symmetry in contingency tables." *JASA* 43(244), 572–574, 1948; Stuart, A. "A test for homogeneity of the marginal distributions in a two-way classification." *Biometrika* 42(3–4), 412–416, 1955; Maxwell, A.E. "Comparing the classification of subjects by two independent judges." *Br. J. Psychiatry* 116(535), 651–655, 1970.

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
