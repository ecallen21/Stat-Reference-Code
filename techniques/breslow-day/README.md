# Breslow–Day Test for OR Homogeneity Across Strata (Reference §8.6)

Given K 2×2 tables (one per stratum), tests

```
H₀: OR_1 = OR_2 = ... = OR_K = OR_common
```

against the alternative that the true odds ratio *varies* across strata (effect modification / interaction). It's the companion test to Cochran–Mantel–Haenszel: CMH pools evidence for a common OR; Breslow–Day checks whether pooling is legitimate.

## Statistic

Under `H₀`, given `OR_MH` (the Mantel–Haenszel estimate of the common OR), each stratum's `a_k` is noncentral hypergeometric with mean `E_k` and variance `V_k`. The test statistic is

```
BD  =  Σ_k (a_k − E_k)² / V_k        ~ χ²_(K−1)
```

`E_k` requires solving a quadratic per stratum for the unique root of `a d / bc = OR_MH` in the feasible range; `V_k` follows from the hypergeometric variance formula.

**Tarone's correction** improves the χ² approximation:

```
BD_Tarone = BD − (Σ_k (a_k − E_k))² / Σ_k V_k
```

`tarone=True` is on by default (matches DescTools / statsmodels).

## When to use vs. Woolf's test

- **Breslow–Day–Tarone**: standard software default; good small-sample behavior; requires the OR_MH solve per stratum.
- **Woolf's test** (in `cochran-mantel-haenszel`): simpler weighted-least-squares chi-square on `log(OR_k)`; needs cell continuity when any cell is zero; used more historically.

Prefer BD–Tarone.

## Files

- `python/breslow_day.py` — from-scratch stat with the per-stratum quadratic solve, Tarone correction, and graceful handling of pathological OR_MH values (statsmodels errors out in these cases). Matches `statsmodels.stats.contingency_tables.StratifiedTable.test_equal_odds` to 4 decimals on well-behaved data.
- `r/breslow_day.R` — from-scratch + `DescTools::BreslowDayTest`.

## Assumptions

- K independent 2×2 tables (one per stratum). No cell needs to be nonzero — the quadratic solve handles zeros, though small strata with zeros produce a fat lower tail in the reference chi-square.
- Interpret as: "*given* MH pooling is being considered, is the assumption of a common OR defensible?"

## Run

```
python techniques/breslow-day/python/breslow_day.py
Rscript techniques/breslow-day/r/breslow_day.R
```

**Refs:** Breslow, N.E. & Day, N.E. *Statistical Methods in Cancer Research, Vol. 1: The Analysis of Case-Control Studies*, IARC Sci. Pub. 32, 1980; Tarone, R.E. "On heterogeneity tests based on efficient scores." *Biometrika* 72(1), 91–95, 1985.

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
