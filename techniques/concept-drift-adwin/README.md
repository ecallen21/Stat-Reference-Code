# Concept-Drift Detection: ADWIN + DDM (Reference Ch 32 MLOps)

**Online, single-pass detectors** on a stream of loss / error
indicators. Distinct from **data drift** (input distribution `p(x)`
changes) — concept drift is a change in `p(y | x)` that shows up as an
error-rate increase.

## ADWIN (Bifet-Gavaldà 2007)

Maintain a variable-length window `W` of recent losses. For every pair
of contiguous sub-windows `W = W₀ || W₁`, test

```
| mean(W₀) − mean(W₁) |  >  ε(δ, n₀, n₁)
```

with a Hoeffding-style `ε` that guarantees false-alarm rate `≤ δ`. On
firing, drop the older sub-window (concept changed there); `W₁` becomes
the new window. Runs in O(log n) amortised per update.

## DDM (Gama 2004)

Track the running error rate `p_i` and its Bernoulli sd
`σ_i = √( p (1 − p) / i )`. Maintain the minimum-so-far `p_min` and
`σ_min`. Two thresholds:

- **Warning** — `p + σ > p_min + 2 σ_min`
- **Drift**   — `p + σ > p_min + 3 σ_min`

Warning triggers a *shadow training* pipeline; drift triggers a full
model swap.

## When to use

- **Online / streaming model** where per-example labels are eventually
  available.
- **Trigger for retraining** — DDM is the standard tripwire.
- **Cheap to run** — both are `O(1)` memory per stream (ADWIN with
  bucketed implementation).

## When NOT to use

- **Labels are not available in production** — see
  `model-monitoring-metrics` and `data-drift-detection` for label-free
  proxies.
- **Very slow drift** — both detectors are tuned for abrupt changes;
  Page-Hinkley works better for gradual drift.

## Files

- `python/concept_drift_adwin.py` — from-scratch `ADWIN` and `DDM`.
  Synthetic stream with error probability `0.10 → 0.40` at `t = 500`.
  **ADWIN first flag at t = 675 (latency 175); DDM drift flag at t = 555
  (latency 55)**; DDM also fired an early false-alarm warning at
  `t = 117` (expected during the warm-up phase before `p_min` stabilises).
- `r/concept_drift_adwin.R` — `drifter` (R); `river.drift.ADWIN / DDM`,
  `scikit-multiflow`, `alibi-detect` (Python).

## Assumptions & caveats

- **False-alarm rate** — the ADWIN `δ` sets it in theory; empirical
  performance depends on stream structure.
- **DDM warm-up** — the `p_min` heuristic drifts during the first
  ~30 samples, producing occasional early warnings.
- **Reset behavior** — DDM resets on drift; ADWIN gradually flushes
  the old window as the split point moves.
- **Adversarial streams** — Page-Hinkley (`page-hinkley`) is more
  robust to slow drift; KSWIN uses KS on a small window.
- **Latency vs false-alarm** — tighten `δ` (or use `3σ` DDM) for
  fewer false alarms at the cost of slower drift detection.

## Related in this repo

- `data-drift-detection` — label-free `p(x)` monitoring.
- `model-monitoring-metrics` — rolling metrics + alerts.
- `sequential-tests`, `cusum` (if present) — the change-point detection
  cousins.
- `active-learning` — trigger for asking for fresh labels after drift.

## Run

```
python techniques/concept-drift-adwin/python/concept_drift_adwin.py
Rscript techniques/concept-drift-adwin/r/concept_drift_adwin.R
```

**Refs:** Bifet, A. & Gavaldà, R. "Learning from time-changing data with adaptive windowing (ADWIN)." *SDM*, 2007; Gama, J. et al. "Learning with drift detection (DDM)." *SBIA*, 2004; Page, E.S. "Continuous inspection schemes (Page-Hinkley)." *Biometrika*, 1954.

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
