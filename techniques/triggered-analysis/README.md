# Triggered Analysis + Conditional Metrics (Reference §44.13)

Deng & Shi (2016). If only a fraction of randomised users actually
experience the treatment (saw the modal, hit the endpoint), keeping
the non-exposed in the analysis **dilutes** the effect toward zero.

## Triggered analysis

Restrict the analysis to **triggered users** — those who could have
been exposed — in **both arms**. Report the effect conditional on
trigger. Requires the trigger to be measurable in both control and
treatment (control-side "counterfactual trigger" or hypothetical
exposure).

## Relation to ITT

`ITT ≈ trigger_prob × CATE_triggered` (under one-sided
non-compliance). The triggered analysis is the CATE; ITT is the
diluted average.

## When to use

- **Server-side feature rollouts** where you can log "would have
  triggered" for both arms.
- **Modal / banner** experiments where control-side counterfactual
  triggers can be logged.
- **Feature flags** with narrow exposure but broad randomisation.

## When NOT to use

- **Post-randomisation subset** without a valid counterfactual
  trigger — biased. Use CACE / IV framework instead
  (`instrumental-variables`).
- **Guardrail metrics** — always report ITT to catch harm outside
  the exposed subset.

## Files

- `python/triggered_analysis.py` — dilution demo. Setup: n=10000,
  30 % trigger rate, true effect among triggered = 0.5. Result:
  **ITT diff = 0.141** (≈ 0.30 × 0.50 dilution), **triggered
  diff = 0.486** — recovers the true CATE.
- `r/triggered_analysis.R` — `stats` subset + `survey::svyglm`,
  custom (R); `scipy.stats`, `statsmodels`, `causalml` (Python).

## Assumptions & caveats

- **Trigger observable in both arms** — this is the critical
  condition; retrofitting a trigger in control is often hard.
- **Report ITT alongside triggered** — regulators and stakeholders
  want to see both.
- **Trigger heterogeneity** — if triggered subset differs by
  covariates from the whole cohort, generalisability is limited.
- **CACE** framework generalises when compliance is imperfect on
  both sides.

## Related in this repo

- `ab-test-fundamentals`, `cuped-variance-reduction` — companions.
- `hte-uplift` — extends triggered analysis to per-user CATE.
- `instrumental-variables` (if present) — CACE / LATE cousin.

## Run

```
python techniques/triggered-analysis/python/triggered_analysis.py
Rscript techniques/triggered-analysis/r/triggered_analysis.R
```

**Refs:** Deng, A. & Shi, X. "Data-driven metric development for online controlled experiments." *KDD*, 2016.

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
