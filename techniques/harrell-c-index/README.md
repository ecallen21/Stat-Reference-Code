# Concordance C-Statistic (Reference §11.6)

Discrimination measure for survival predictors — the survival analog of AUC. Answers: **given a randomly selected pair where subject `i` had the event before subject `j`, how often is `i`'s predicted risk higher than `j`'s?**

## Harrell C (Harrell 1982)

```
C = concordant pairs / usable pairs
```

- **Usable pair**: the subject with the shorter observation time experienced the event.
- **Concordant**: the shorter-time subject also has the higher predicted risk.

Range `[0, 1]`; `0.5` = random; `1.0` = perfect ranking.

## Uno IPCW C (Uno et al. 2011)

Harrell C is **biased under heavy or dependent censoring**. Uno's IPCW-corrected version uses the estimated censoring distribution `G(t) = P(C > t)` as weights and is consistent under general censoring:

```
C_uno = Σ_{(i, j)}  I[T_i < T_j, δ_i = 1, T_i ≤ τ]  I[pred_i > pred_j]  · w_i
      / Σ_{(i, j)}  I[T_i < T_j, δ_i = 1, T_i ≤ τ]  · w_i
w_i = 1 / Ĝ(T_i)²
```

Report **both** Harrell and Uno when censoring is heavy or you suspect it depends on covariates.

## Files

- `python/harrell_c_index.py` — from-scratch pairwise Harrell C + Uno IPCW variant with KM-based censoring estimator. Demo (n = 300, true linear predictor): Harrell C = 0.68, Uno C = 0.67 (informative); random predictor C ≈ 0.47.
- `r/harrell_c_index.R` — `survival::concordance` (canonical) and `Hmisc::rcorr.cens`.

## When to use

- **Any survival model** — Cox, parametric, random-forest survival — needs a discrimination summary.
- **Model comparison** on the same dataset: ΔC between candidate risk scores.
- **External validation** — apply the risk score to a new cohort and report C.

## Interpretation cheat sheet

| C value | Interpretation                                    |
|---------|----------------------------------------------------|
| ≥ 0.85  | Excellent — likely too good; check for leakage.   |
| 0.75 – 0.85 | Good discrimination for a clinical model.     |
| 0.65 – 0.75 | Modest; typical of most Cox models.           |
| ≤ 0.60  | Weak; consider more features or a better model.   |

## Assumptions & caveats

- **Non-informative censoring** for Harrell C to be unbiased. Under dependent censoring, use Uno IPCW.
- **Time-varying predictors**: extend to time-varying C-index (`survC1::Est.Cval`).
- **Complement**: report a **calibration** measure too (calibration slope + intercept, calibration-in-the-large). Good discrimination does not imply good calibration.

## Run

```
python techniques/harrell-c-index/python/harrell_c_index.py
Rscript techniques/harrell-c-index/r/harrell_c_index.R
```

**Refs:** Harrell, F.E. et al. "Evaluating the yield of medical tests." *JAMA* 247(18), 2543–2546, 1982; Uno, H. et al. "On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data." *Stat. Med.* 30(10), 1105–1117, 2011; Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015.

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
