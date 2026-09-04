# Polygenic Risk Scores (Reference §40.2, §40.15)

Choi-Mak-O'Reilly (2020), Wand et al. (2021). A polygenic risk score
(PRS) aggregates the small effects of many variants into a single
per-individual sum:

```
PRS_i = Σ_j β̂_j · G_ij
```

## Pruning + thresholding (P+T)

1. Run a discovery-cohort **GWAS** → per-SNP `β̂_j`, `p_j`.
2. **LD-prune** to keep near-independent SNPs (`r² < 0.1`).
3. **Threshold** on `p` (5 × 10⁻⁸ strict; 10⁻³, 10⁻¹, 1 inclusive)
   and sum weighted genotypes in a target cohort.

Modern variants (**LDpred**, **PRS-CS**, **lassosum**) do continuous
shrinkage instead of hard thresholding.

## Evaluation

- **Discrimination** — AUC (binary outcome) or R² (continuous).
- **Incremental value** — add PRS to a clinical baseline; report
  ΔAUC / ΔR².

## Files

- `python/polygenic_risk_scores.py` — P+T at five thresholds
  computed on a target cohort using discovery-GWAS effect sizes;
  incremental R² on top of a clinical baseline. Demo (n_disc=3000,
  n_targ=1000, 50 true causal SNPs of β~N(0, 0.05²)): **ΔR²
  optimised at p<0.1 (0.0153)** with 87 SNPs, better than strict
  p<5e-8 (ΔR² 0.008).
- `r/polygenic_risk_scores.R` — `bigsnpr`, `lassosum` (R);
  `ldpred`, PRSice2 external (Python).

## When to use

- **Risk stratification** — flag high-genetic-risk individuals for
  screening (breast cancer, cardiovascular disease).
- **Discovery** — orthogonal predictor to clinical variables.

## When NOT to use

- **Trans-ancestry deployment** without care — PRS derived from
  European cohorts underperform in other ancestries and exacerbate
  disparities (Martin et al. 2019).
- **Individual prediction** without recalibration — PRS quantiles
  work for populations, not for firm individual thresholds.

## Assumptions & caveats

- **LD structure identical** between discovery and target
  populations — violated across ancestries.
- **Sample overlap** between discovery and target inflates PRS
  performance (use separate cohorts).
- **Absolute risk** requires calibration to the target-population
  prevalence.
- **Report ancestry, cohort ascertainment, and thresholds** per
  the PGS Catalog reporting standards (Wand 2021).

## Related in this repo

- `gwas` — where the `β̂_j` come from.
- `linkage-disequilibrium` — the LD pruning step.
- `clinical-risk-scores`, `discrimination-calibration`,
  `bootstrap-optimism-correction` — the evaluation stack.

## Run

```
python techniques/polygenic-risk-scores/python/polygenic_risk_scores.py
Rscript techniques/polygenic-risk-scores/r/polygenic_risk_scores.R
```

**Refs:** Choi, S.W., Mak, T.S.-H., & O'Reilly, P.F. "Tutorial: a guide to performing polygenic risk score analyses." *Nature Protocols*, 2020; Wand, H., Lambert, S.A., Tamburro, C. et al. "Improving reporting standards for polygenic scores in risk prediction studies." *Nature*, 2021.

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
