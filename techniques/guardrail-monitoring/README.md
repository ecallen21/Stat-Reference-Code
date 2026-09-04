# Guardrail Monitoring (Reference §44.15)

Kohavi-Tang-Xu (2020), Fabijan et al. (2017). Guardrail metrics
track **harm** — latency, crash rate, revenue drop, core-experience
regression. Halt the experiment when a guardrail crosses a defined
threshold.

## Sequential monitor

- **Wilson CI** on the running treatment proportion, refreshed
  per batch.
- **Alert** when `CI_lo > baseline + tolerance` (harm confirmed) —
  use a very small α (e.g., 0.001) to control false alarms across
  many peeks.
- **Stop-for-harm** — pull the treatment on alert.

## When to use

- **Every production A/B rollout** — guardrails run alongside
  primary metric monitoring.
- **Feature flags** with kill switches.

## When NOT to use

- **Guardrails with unknown baseline distribution** — calibrate
  first on a control-only period.

## Files

- `python/guardrail_monitoring.py` — Wilson CI + streaming
  guardrail monitor. Demo (baseline 0.5 %, tolerance 0.5 %,
  α=0.001): Case A (guardrail stays at baseline) → no alert;
  Case B (regresses to 2 %) → **alert at batch 2, n=3000, rate
  1.6 %, CI (1.0 %, 2.6 %)**.
- `r/guardrail_monitoring.R` — `stats::binom.test` /
  `prop.test`, `qcc`, `Hmisc::binconf` (R);
  `scipy.stats.beta.ppf`, `statsmodels.stats.proportion`,
  commercial SDKs (Python).

## Assumptions & caveats

- **Multiple guardrails** need multiple-comparison correction; use
  a family-wise α cap or hierarchical procedure.
- **Effect-size cutoff** (tolerance) is a policy choice — clinical
  vs statistical significance.
- **Same-user re-triggering** biases repeated observations —
  cluster on user.
- **Ethical stopping** — guardrail alarms bind: the launch is
  paused pending investigation.

## Related in this repo

- `always-valid-inference` — companion sequential inference.
- `ab-test-fundamentals`, `multiple-metrics-fdr` — primary-metric
  analysis.

## Run

```
python techniques/guardrail-monitoring/python/guardrail_monitoring.py
Rscript techniques/guardrail-monitoring/r/guardrail_monitoring.R
```

**Refs:** Kohavi, R., Tang, D., & Xu, Y. *Trustworthy Online Controlled Experiments*, Cambridge University Press, 2020; Fabijan, A., Dmitriev, P., Olsson, H.H., & Bosch, J. "The evolution of continuous experimentation in software product development." *ICSE*, 2017.

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
