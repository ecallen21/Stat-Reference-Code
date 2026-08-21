# NRI and IDI (Reference §20.x extra)

**Net Reclassification Improvement** and **Integrated Discrimination
Improvement** (Pencina et al. 2008) compare an **old** risk model to a **new**
risk model in a binary-outcome cohort.

## Continuous NRI

```
NRI = P(p_new > p_old | event)   − P(p_new < p_old | event)
    + P(p_new < p_old | non-event) − P(p_new > p_old | non-event)
```

For events, movement UP is good; for non-events, movement DOWN is good.
NRI is in [−2, +2]; positive → new model reclassifies correctly on net.

## Category-based NRI

Bin `p_old` and `p_new` at a set of clinically meaningful thresholds (e.g.
10-year risk categories 0–5%, 5–10%, 10–20%, > 20%). Count movements
between categories rather than any change in probability.

## IDI

```
IDI = (mean(p_new | event) − mean(p_old | event))
    − (mean(p_new | non-event) − mean(p_old | non-event))
```

Roughly: how much the new model increases separation between the event and
non-event distributions.

## When to use

- **Model comparison** in clinical / actuarial prediction — should we add this new biomarker?
- **Guideline changes** — does redistricting risk categories reclassify patients meaningfully?
- **Discrimination + clinical utility** — pair with `roc-auc-analysis` (ΔAUC / DeLong test) and `decision-curve-analysis`.

## Files

- `python/nri_idi.py` — continuous NRI, category-based NRI, IDI. Demo (n=2000, logistic outcome, old model uses only x1, new adds x2 which is a true predictor): continuous NRI +0.60, categorical NRI +0.43 at cutoffs [0.10, 0.30, 0.60], IDI +0.117, ΔAUC +0.107 — new model clearly improves.
- `r/nri_idi.R` — `Hmisc::improveProb`, `nricens::nribin / nricens`, `survIDINRI` for survival outcomes.

## Assumptions & caveats

- **NRI is controversial** — Pepe (2011), Kerr et al. (2014) show that continuous NRI can be positive for a model that adds *no* information; category-based NRI depends heavily on where thresholds are set. Report NRI **with** ΔAUC, IDI, and decision-curve analysis.
- **Miscalibration** — NRI/IDI reward a *shift* in predicted probabilities even without calibration; always check calibration (see `calibration-scaling`) before interpreting.
- **Standard errors** — bootstrap (Pencina's asymptotic formulas overstate precision).
- **Class imbalance** — the event proportion determines the scale; report event and non-event components separately.
- **Categorical NRI is preferred** when clinical actions depend on discrete risk-category thresholds.
- **Survival-outcome NRI** requires event-time reweighting (`survIDINRI`, `nricens::nricens`); do not use the binary formulas on censored data.

## Related in this repo

- `roc-auc-analysis` — AUC + DeLong test for pairwise comparison.
- `calibration-scaling` — calibration slope / intercept, Brier decomposition.
- `decision-curve-analysis` — net-benefit view of model comparison across thresholds.
- `harrell-c-index` — survival-outcome discrimination.

## Run

```
python techniques/nri-idi/python/nri_idi.py
Rscript techniques/nri-idi/r/nri_idi.R
```

**Refs:** Pencina, M.J. et al. "Evaluating the added predictive ability of a new marker: from area under the ROC curve to reclassification and beyond." *Stat. Med.* 27(2), 157–172, 2008; Pepe, M.S. et al. "Testing for improvement in prediction model performance." *Stat. Med.* 32(9), 1467–1482, 2013; Kerr, K.F. et al. "Net reclassification indices for evaluating risk prediction instruments: A critical review." *Epidemiology* 25(1), 114–121, 2014.

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
