# Canary Deployment (Reference Ch 32 MLOps)

**Roll a candidate model in progressively** — 5 % → 25 % → 50 % → 100 %
of traffic — while monitoring SLOs at each stage and rolling back
automatically on violation.

## Standard schedule

| Stage | Traffic to candidate | Bake time    | Metric checkpoint            |
|------:|---------------------:|:-------------|:-----------------------------|
| 1     | 5 %                  | 30-60 min    | error rate, p99 latency      |
| 2     | 25 %                 | 1-2 hr       | + downstream business metric |
| 3     | 50 %                 | 2-4 hr       | + subgroup fairness metrics  |
| 4     | 100 %                | steady state | full monitoring              |

## Rollback rule

Any SLO violation at stage `k` → traffic reverts to stage `k − 1` and
the candidate is either fixed and re-canaried, or abandoned. A
persistent violation after `max_rollbacks` triggers **hold at stable
stage** and paging.

## When to use

- **Any production model swap** — even minor version bumps.
- **Serving infrastructure supports weighted traffic** (Istio,
  service-mesh, feature-flag SDK).
- **Real-time labels** or business metrics (revenue, click) — otherwise
  see `shadow-deployment` first.

## When NOT to use

- **No traffic-splitting layer** — batch-mode / offline scoring can't
  be canaried; rely on `shadow-deployment` before promotion.
- **Very sparse label signal** — the SLO metrics take too long to be
  meaningful at each stage.
- **Heavily stateful workflows** (session personalisation) — canary
  requires per-user consistency.

## Files

- `python/canary_deployment.py` — from-scratch `CanaryRouter` with
  Bernoulli traffic assignment, per-stage SLO checks on error rate +
  p99 latency, auto-rollback + `max_rollbacks` guard. Demo: candidate
  with a regression bug on `x[0] > 0.5`; router advances through 5 %
  → 25 % → 50 % (all OK), fails at 100 % (err = 0.33), rolls back to
  50 %, retries 100 %, fails again, holds at 50 % (stable).
- `r/canary_deployment.R` — `plumber` / `vetiver` (R);
  `seldon-core`, `kserve`, `argo-rollouts`, `Istio VirtualService`
  (Python / Kubernetes ecosystem).

## Assumptions & caveats

- **Stage lengths matter** — too short = noisy SLO estimates; too long
  = slow rollout. Match to your metric's convergence timescale.
- **Traffic assignment** — Bernoulli sampling is stateless (simple);
  hash-based user assignment gives per-user consistency.
- **SLO thresholds** must be pre-committed — the moment a decision
  hinges on "what if we lower the bar", you have already lost.
- **Rollback vs abort** — repeated rollback loops indicate the
  candidate is bad; use `max_rollbacks` (default 2) to avoid oscillation.
- **Interleaved-canary** (Kohavi 2020) is a variance-reduction
  alternative for A/B-style feature comparison.

## Related in this repo

- `shadow-deployment` — the risk-free predecessor to canary.
- `model-monitoring-metrics` — the metric machinery under each stage.
- `bayesian-ab-testing`, `multi-armed-bandits` — decide when to
  promote to 100 %.
- `experiment-tracking`, `model-registry-versioning` — where the
  canary's promotion / rollback is recorded.

## Run

```
python techniques/canary-deployment/python/canary_deployment.py
Rscript techniques/canary-deployment/r/canary_deployment.R
```

**Refs:** Chip Huyen. *Designing Machine Learning Systems*, O'Reilly, 2022 (ch. 9); Kohavi, R., Tang, D. & Xu, Y. *Trustworthy Online Controlled Experiments*, Cambridge University Press, 2020.

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
