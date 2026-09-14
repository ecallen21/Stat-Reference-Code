# Temperature Scaling (Reference §47.281)

Guo, Pleiss, Sun & Weinberger (2017). Post-hoc calibration for
deep networks: divide LOGITS by a single scalar `T > 0` before
softmax:

```
p_calibrated = softmax(z / T)
```

Fit `T` by minimising NLL on a held-out validation set. Only
1 parameter — cheap and expressive enough to fix confidence
miscalibration of most classifiers WITHOUT changing predicted
classes.

## Files

- `python/temperature_scaling.py` — grid + scipy `minimize_scalar`.
  Demo: simulated overconfident 5-class classifier (n=2000).
  ECE 0.09 → 0.01 after T=0.97 scaling; accuracy unchanged
  because argmax is preserved.
- `r/temperature_scaling.R` — `probably`, custom `optim`,
  `torch` (R); `netcal.scaling.TemperatureScaling`,
  `torchcalibration`, from-scratch (Python).

## When to use

- **Deep classifiers** — Guo et al showed T-scaling fixes ECE
  on most CNN / Transformer classifiers.
- **Post-hoc, minimal-touch** — no retraining, only 1 param.
- **When argmax must be preserved** — T > 0 leaves the
  predicted class unchanged.

## When NOT to use

- **Systematic per-class bias** — a single T can't fix
  class-dependent miscalibration; use vector / matrix scaling.
- **Extreme miscalibration** — non-monotone reliability
  curves need isotonic / histogram binning.
- **Very small validation set** (< 200) — T is noisy.

## Assumptions & caveats

- **Class-conditional miscalibration** — vanilla T-scaling
  assumes uniform overconfidence across classes.
- **Optimisation** — 1-D scalar; grid + Brent is safe.
- **Reuse the same T** across all downstream inference.
- **Reports** — always show ECE before/after AND accuracy
  unchanged.

## Related in this repo

- `platt-scaling` — binary-specific analogue.
- `beta-calibration`, `histogram-binning-calibration` —
  richer post-hoc alternatives.
- `expected-calibration-error` — the metric optimised.
- `model-recalibration`, `calibration-plots` — visual /
  companion tools.

## Run

```
python techniques/temperature-scaling/python/temperature_scaling.py
Rscript techniques/temperature-scaling/r/temperature_scaling.R
```

**Refs:** Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. "On calibration of modern neural networks." In *ICML*, pp. 1321-1330, 2017.

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
