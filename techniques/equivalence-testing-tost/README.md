# Equivalence Testing via TOST (Reference §3.21)

A classic t-test asks **"are the means different?"** — it can only reject difference; it can *never* "prove" equality. **TOST** (Two One-Sided Tests) flips this: you commit to an **equivalence margin** `[lower, upper]` (the smallest difference you'd care about), and reject the hypothesis "the means differ by more than the margin" if **both** one-sided tests at level α are significant.

## The math

Per side:
- Lower test: `H₀_L: diff ≤ lower` vs. `H₁_L: diff > lower`  → `t_L = (diff − lower) / SE`
- Upper test: `H₀_U: diff ≥ upper` vs. `H₁_U: diff < upper`  → `t_U = (diff − upper) / SE`

Reject equivalence's `H₀` (i.e. **declare equivalence**) iff both p-values are below α. The TOST p-value is `max(p_L, p_U)`.

## The 90% CI rule

At `α = 0.05`, "TOST rejects" ↔ "the `100·(1 − 2α)%` = **90% CI** for the difference lies *entirely inside* `[lower, upper]`". Reporting the CI is at least as informative as the p-value here.

## Choosing the margin

This is a *domain* decision, not a statistical one. Common conventions:
- **Bioequivalence** (drug exposure ratios): `[80%, 125%]` on the log-transformed ratio of geometric means.
- **Behavioral / educational**: Cohen's `d` in `[−0.2, 0.2]` ("smallest effect size of interest").
- **Method comparison (assays, devices)**: a clinically meaningful tolerance in original units.

Picking the margin *after* seeing the data invalidates the test (the same flavor of "garden of forking paths" that affects p-values).

## Variants implemented
- **One-sample TOST** — is `mean(x)` within `[lower, upper]`?
- **Two-sample TOST** — Welch by default (recommended), Student available; same sign convention as `techniques/t-tests`.
- **Paired TOST** — one-sample TOST on the differences `x₁ − x₂` (e.g. equivalence of pre vs. post when you *want* no change).

## Files
- `python/tost.py` — from-scratch one-sample / two-sample (Welch/Student) / paired TOST; compares against `statsmodels.stats.weightstats.ttost_ind`.
- `r/tost.R` — from-scratch + `equivalence::tost`; notes on `TOSTER::TOSTtwo` etc.
- PySpark: N/A — TOST is closed-form once the per-group `n / mean / var` are known; reuse the aggregation from `techniques/t-tests/pyspark/` and apply TOST on the driver.

## Run
```
python techniques/equivalence-testing-tost/python/tost.py
Rscript techniques/equivalence-testing-tost/r/tost.R
```

**Refs:** Schuirmann, "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability," *J. Pharmacokinetics and Biopharmaceutics* 15(6), 657–680, 1987; Lakens, "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses," *Social Psychological and Personality Science* 8(4), 355–362, 2017.
