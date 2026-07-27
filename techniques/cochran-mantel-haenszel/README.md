# Cochran–Mantel–Haenszel Test + MH Common OR (Reference §8.3, §8.16)

The **stratified** analogue of the χ² test on a 2×2 table. Given K independent 2×2 tables (one per stratum: age band, site, matched set, …), CMH pools the evidence for a common exposure–outcome association *conditional on the stratum*.

## The setup

For each stratum k:

```
              outcome +   outcome −
     expo +    a_k         b_k        n1_k
     expo −    c_k         d_k        n0_k
              m1_k        m0_k        N_k
```

`H₀`: within every stratum, exposure and outcome are independent (no partial association).

## CMH statistic

Under `H₀`, `a_k` is hypergeometric with mean `E(a_k) = n1_k · m1_k / N_k` and variance `Var(a_k) = n1_k · n0_k · m1_k · m0_k / (N_k² (N_k − 1))`. Sum over strata:

```
        [Σ_k (a_k − E(a_k))]²
X²_CMH = ─────────────────── ~ χ²₁ under H₀
             Σ_k Var(a_k)
```

The optional Mantel-Haenszel continuity correction subtracts 0.5 from `|Σ(a − E)|`.

## Mantel–Haenszel common OR

```
         Σ_k a_k · d_k / N_k
OR_MH = ──────────────────
         Σ_k b_k · c_k / N_k
```

The **Robins–Breslow–Greenland** (RBG) SE for `log(OR_MH)` is what every statistics package reports; formulas in the code. Wald 95% CI on the log scale, then exponentiated.

## Woolf's test (homogeneity)

`H₀`: the K stratum ORs are equal. Inverse-variance-weighted χ² on `log(OR_k)` around their weighted mean, `df = K − 1`. See the dedicated `breslow-day` technique for the more standard homogeneity test.

## Files

- `python/cochran_mantel_haenszel.py` — CMH statistic (with/without continuity), MH OR + RBG CI, Woolf's homogeneity. Matches `statsmodels.stats.contingency_tables.StratifiedTable` for the CMH statistic and OR_MH to 12 decimals.
- `r/cochran_mantel_haenszel.R` — from-scratch + base `stats::mantelhaen.test`.
- `pyspark/cochran_mantel_haenszel.py` — `groupBy(stratum, exposure, outcome).count()` builds the K 2×2 tables when the raw data has one row per subject; drives the driver-side scalar computation.

## Run

```
python techniques/cochran-mantel-haenszel/python/cochran_mantel_haenszel.py
Rscript techniques/cochran-mantel-haenszel/r/cochran_mantel_haenszel.R
python techniques/cochran-mantel-haenszel/pyspark/cochran_mantel_haenszel.py
```

**Refs:** Mantel, N. & Haenszel, W. "Statistical aspects of the analysis of data from retrospective studies of disease." *JNCI* 22(4), 719–748, 1959; Cochran, W.G. "Some methods for strengthening the common χ² tests." *Biometrics* 10(4), 417–451, 1954; Robins, J., Breslow, N. & Greenland, S. "Estimators of the Mantel-Haenszel variance consistent in both sparse-data and large-strata limiting models." *Biometrics* 42(2), 311–323, 1986.

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
