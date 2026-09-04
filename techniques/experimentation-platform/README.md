# Experimentation Platform Primitives (Reference §44.11)

Kohavi-Tang-Xu (2020 ch 3-5, 22). Production A/B infrastructure
requires a small set of primitives to run experiments safely at
scale.

## Core primitives

- **Deterministic assignment** — hash `(experiment_name, user_id)`
  → variant; identical for the same user across sessions.
- **Namespaces / layers** — concurrent experiments must not
  collide; Facebook's PlanOut coordinates layers.
- **Sample Ratio Mismatch (SRM) check** — χ² test that observed
  traffic matches designed split; a small p (< 0.001) is a bug
  alarm.
- **Interaction detection** — cross-experiment analysis to catch
  variants that only differ when combined.

## When to use

- **Every production A/B platform** — SRM in particular should be
  automated and blocking before any metric is trusted.

## When NOT to use

- **One-off pilots** where hand-checking is cheaper than
  infrastructure.

## Files

- `python/experimentation_platform.py` — MD5-based deterministic
  assignment + SRM χ² check. Demo (100 000 users, 50/50 planned):
  actual 49689/50311, χ²=3.87, p=0.049 → **not SRM** (borderline
  because χ² is sensitive at large n; the SRM threshold is 0.001);
  buggy 45000/55000 case: χ²=1000, p≈0 → **SRM detected**.
  Different namespaces decouple the same user across experiments.
- `r/experimentation_platform.R` — `stats::chisq.test`, `digest`,
  `pwr` (R); `planout`, `scipy.stats`, commercial SDKs
  (`eppo-sdk`, `statsig`, `growthbook`) (Python).

## Assumptions & caveats

- **Hash quality** — use a wide, uniformly-distributed hash
  (MD5 / MurmurHash3 / SipHash); avoid `random.random()`
  because it changes with session state.
- **SRM threshold** — Kohavi recommends 0.001 (very conservative)
  to avoid noise-triggered alarms.
- **Independence across namespaces** — different experiment names
  must produce independent hashes for the same user.
- **Interaction detection** at scale needs pairwise ANOVA / factor
  analysis; the platform layer usually only flags "top-K" pairs.

## Related in this repo

- `ab-test-fundamentals`, `mde-sample-size` — the analytic layer
  on top of the platform.
- `experiment-tracking`, `model-registry-versioning` — MLOps
  cousins.

## Run

```
python techniques/experimentation-platform/python/experimentation_platform.py
Rscript techniques/experimentation-platform/r/experimentation_platform.R
```

**Refs:** Kohavi, R., Tang, D., & Xu, Y. *Trustworthy Online Controlled Experiments*, Cambridge University Press, 2020 (ch 3-5, 22); Fabijan, A., Dmitriev, P., Olsson, H.H., & Bosch, J. "The evolution of continuous experimentation in software product development." *ICSE*, 2017.

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
