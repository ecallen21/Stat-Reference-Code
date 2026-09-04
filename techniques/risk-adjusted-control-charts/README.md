# Risk-Adjusted Control Charts (Reference §37.10)

Steiner et al. (2000), Lovegrove et al. (1997). Healthcare quality
monitoring where outcomes (mortality, readmission) depend on **patient
mix**. Standardise via a predictive risk model and monitor deviations.

## Two workhorses

### VLAD (Variable Life-Adjusted Display)

```
V_i = Σ_{j ≤ i} ( p̂_j − y_j ).
```

Downward slope = worse-than-expected outcomes; upward = better.

### Risk-adjusted CUSUM (Steiner 2000)

Log-likelihood-ratio CUSUM for detecting an **odds-ratio shift** OR:

```
W_i = y_i · log(OR) − log(1 − p̂_i + p̂_i · OR)
L_i^+ = max(0, L_{i−1}^+ + W_i)
```

Signal when `L^+ > h`.

## When to use

- **Surgeon / provider monitoring** in cardiac surgery, obstetrics,
  ICU.
- **Post-market drug / device safety** with baseline risk model.
- **Any binary-outcome monitoring** with patient-mix confounding.

## When NOT to use

- **No reliable risk model** — plain CUSUM is more transparent.
- **Rare outcomes** — use Bernoulli CUSUM (see `rare-event-control-
  charts`).

## Files

- `python/risk_adjusted_control_charts.py` — VLAD + Steiner CUSUM.
  Demo: 200 patients, baseline risk 5-30 %, odds ratio doubles at
  patient 100. VLAD **+2.71 at n=100 → −10.02 at n=200** (visible
  drop); CUSUM signals at t=169 (delay 69) with h=4.5.
- `r/risk_adjusted_control_charts.R` — `vlad`, `runstats` (R); custom
  Python.

## Assumptions & caveats

- **Risk model quality** — mis-calibration inflates false alarms;
  refit periodically.
- **OR choice** — larger OR = fast detection of large shifts, slow
  for small ones.
- **h tuning** — target ARL_0 = 10 000 patients (a year of surgery)
  typical for healthcare.
- **VLAD is descriptive, not a hypothesis test** — pair with CUSUM
  for formal alarm.
- **Multiple surgeons** — apply separately; monitor pooled with
  hierarchical adjustments.

## Related in this repo

- `cusum-charts`, `ewma-charts`, `shewhart-control-charts` — non-
  risk-adjusted parents.
- `rare-event-control-charts` — Bernoulli CUSUM for rare events.
- `calibration-scaling` — for the risk-model input.
- `logistic-regression`, `cox-ph` — typical risk-model families.

## Run

```
python techniques/risk-adjusted-control-charts/python/risk_adjusted_control_charts.py
Rscript techniques/risk-adjusted-control-charts/r/risk_adjusted_control_charts.R
```

**Refs:** Lovegrove, J. et al. "Monitoring the results of cardiac surgery." *Lancet*, 1997; Steiner, S.H. et al. "Monitoring surgical performance using risk-adjusted cumulative sum charts." *Biostatistics*, 2000; Grigg, O. & Farewell, V. "An overview of risk-adjusted charts." *JRSS-A*, 2004.

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
