# Permutation / Randomization Tests (Reference §10.7; also covers §10.16)

Under `H₀`, group labels (or one variable in a correlation, or a response given a fixed X) are **exchangeable** with the outcome. Randomly permute the labels many times; recompute the test statistic on each shuffle; compare the observed statistic to the empirical null distribution.

## Test statistics implemented

| Test | Statistic | Permutation |
|---|---|---|
| **Two-sample** | `mean(A) − mean(B)` (or any user function) | Shuffle group labels |
| **Correlation** | Pearson `r(x, y)` | Permute `y` (or `x`) |
| **Regression coef** | `β̂_k` on the observed X | Permute `y` (full-response); tests joint null across all coefs |

For a coefficient test that isolates the effect of predictor k while holding the others fixed, use **Freedman–Lane** residual permutation — not implemented here; can be built on the wild-bootstrap machinery.

## Exact vs. Monte-Carlo

- **Exact**: enumerate all `n! / (n₁! n₂!)` label assignments. Feasible only for tiny samples (< ~15).
- **Monte-Carlo** (default): sample `B` random permutations. Unbiased for any `B`; smaller `B` just widens the p-value's uncertainty.

## p-value formula (Phipson–Smyth add-1)

```
p̂ = (1 + #{|T_perm| ≥ |T_obs|}) / (1 + B)
```

The add-1 avoids a p-value of exactly 0 when no permutation exceeds the observed statistic (which would be biologically implausible — you *did* observe the data at some tiny probability).

## Files

- `python/permutation_tests.py` — two-sample, correlation, and regression-coef permutation tests, with cross-check against `scipy.stats.permutation_test`.
- `r/permutation_tests.R` — from-scratch versions; optional `coin` library for `oneway_test` / `spearman_test`.
- `pyspark/permutation_tests.py` — observed diff computed via Spark `groupBy`; permutation loop on the driver (would sample the column for truly huge n).

## Assumptions

- **Exchangeability under `H₀`**. That's it — no distributional assumption on the data.
- For two-sample tests with unequal variances, permutation tests inherit the group-variance structure — a significant result may reflect a scale difference, not a mean shift. Interpret as "the two distributions differ" unless you also equalize variances.

## Run

```
python techniques/permutation-tests/python/permutation_tests.py
Rscript techniques/permutation-tests/r/permutation_tests.R
python techniques/permutation-tests/pyspark/permutation_tests.py
```

**Refs:** Fisher, R.A. *The Design of Experiments*, Oliver & Boyd, 1935; Pitman, E.J.G. "Significance tests which may be applied to samples from any populations." *JRSS Suppl.* 4(1), 119–130, 1937; Good, P.I. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed., Springer, 2005; Phipson, B. & Smyth, G.K. "Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn." *Stat. Appl. Genet. Mol. Biol.* 9(1), Art. 39, 2010.

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
