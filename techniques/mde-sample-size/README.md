# Sample Size + Minimum Detectable Effect (Reference §44.2)

Kohavi-Tang-Xu (2020 ch 14-15). Planning-time calculations for
an A/B test. Two forward-and-reverse formulations:

- Given `(baseline, MDE, α, power)` → **n per arm**.
- Given `(baseline, n per arm, α, power)` → **MDE**.

## Formulas

**Two-proportion**

```
n_per_arm = ((z_{α/2} √(2 p̄ q̄) + z_β √(p_C q_C + p_T q_T))² ) / δ²
```

**Two-sample continuous**

```
n_per_arm = 2 (z_{α/2} + z_β)² σ² / δ²
```

## When to use

- **Pre-experiment planning** — traffic and runtime budget.
- **Post-experiment sanity** — was the run large enough to detect
  a plausible effect?

## When NOT to use

- **Sequential / adaptive designs** — closed-form formulas over-
  estimate required n; use group-sequential or always-valid tables.
- **Ratio / clustered / network** metrics — use variance-inflation-
  factor-adjusted formulas.

## Files

- `python/mde_sample_size.py` — closed-form n and MDE for
  proportions + t-tests. Demo: `p=0.05, MDE=0.005` → **n=31 234
  per arm**; halving MDE to 0.001 blows n up to **752 703**. For
  a continuous metric with σ=1, MDE=0.05 → **n=6 280 per arm**.
- `r/mde_sample_size.R` — `pwr::pwr.t.test`/`pwr.2p.test`,
  `WebPower`, `stats::power.t.test`/`power.prop.test` (R);
  `statsmodels.stats.power.TTestIndPower` (Python).

## Assumptions & caveats

- **Variance / baseline rate estimate** dominates n; use recent
  data or pilots.
- **α, power convention** — 0.05, 0.80 usual; some orgs pick 0.90.
- **Two-tailed vs one-tailed** — two-tailed is safer.
- **Multiple metrics** — Bonferroni or FDR corrections need larger
  n per test.

## Related in this repo

- `ab-test-fundamentals` — the analysis; MDE plans it.
- `cuped-variance-reduction` — shrinks n by reducing σ.
- `always-valid-inference` — sequential-safe sample sizing.

## Run

```
python techniques/mde-sample-size/python/mde_sample_size.py
Rscript techniques/mde-sample-size/r/mde_sample_size.R
```

**Refs:** Kohavi, R., Tang, D., & Xu, Y. *Trustworthy Online Controlled Experiments*, Cambridge University Press, 2020 (ch 14-15); van Belle, G. *Statistical Rules of Thumb*, 2nd ed., Wiley, 2008.

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
