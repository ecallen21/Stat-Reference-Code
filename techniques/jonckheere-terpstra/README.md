# Jonckheere–Terpstra Test (Reference §6.10)

A Kruskal–Wallis-style test against an **ordered alternative**: H_A says the group distributions are stochastically ordered `F₁ ⪯ F₂ ⪯ … ⪯ F_k` (or the reverse). When the ordering is theoretically motivated (low / medium / high dose, increasing exposure, calendar year), JT is **more powerful** than the unordered K–W.

## Statistic

`J = Σᵢ<ⱼ Uᵢⱼ`, where `Uᵢⱼ` is the Mann–Whitney U between groups `i` and `j` (with ties contributing 0.5).

Under H₀ (all groups identical):
- `E[J] = (N² − Σnᵢ²) / 4`
- `Var[J] = (N²(2N+3) − Σnᵢ²(2nᵢ+3)) / 72`
- `z = (J − E[J]) / √Var[J]` ~ standard normal (asymptotic)

## When to use

- Groups have a **natural ordering** baked into the design (dose, age band, severity).
- You want to test "is there a *monotone trend* across the levels?" rather than the weaker "are any two different?"

If you do an unordered K–W and *then* eyeball the result for ordering, you're effectively testing what JT would test — but with worse power and an unclear inferential status.

## Files
- `python/jonckheere_terpstra.py` — from-scratch Mann-Whitney pair counts + normal-approximation z; library cross-check via `scipy.stats.jonckheere` (scipy 1.13+; falls back to a note otherwise).
- `r/jonckheere_terpstra.R` — from-scratch + `clinfun::jonckheere.test` / `DescTools::JonckheereTerpstraTest`.
- PySpark: N/A (small-sample test).

## Run
```
python techniques/jonckheere-terpstra/python/jonckheere_terpstra.py
Rscript techniques/jonckheere-terpstra/r/jonckheere_terpstra.R
```

**Refs:** Jonckheere, "A Distribution-Free k-Sample Test against Ordered Alternatives," *Biometrika* 41(1/2), 133–145, 1954; Terpstra, "The Asymptotic Normality and Consistency of Kendall's Test against Trend," *Indagationes Math.* 14, 327–333, 1952.
