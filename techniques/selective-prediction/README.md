# Selective Prediction / Abstention (Reference Ch 29 UQ)

Give the model a **third choice**: predict, or refuse to answer. Chow
(1957) first formulated the risk-coverage trade-off; Geifman & El-Yaniv
(2017) revived it for deep networks.

## The framework

- Base classifier `f(x)` — top-1 label.
- Confidence score `g(x) ∈ [0, 1]` — max softmax, predictive entropy,
  MC-dropout variance, evidential vacuity, conformal set size, distance
  to training features, etc.
- **Rule**: predict `f(x)` iff `g(x) ≥ τ`; otherwise **abstain**.

## Metrics

```
coverage       c(τ)     = P( g(x) ≥ τ )
selective err  e(τ)     = P( f(x) ≠ y | g(x) ≥ τ )
selective acc           = 1 − e(τ)
```

**Risk-coverage curve** = `(c(τ), e(τ))` swept over `τ`.
**AURC** (Area Under Risk-Coverage) — single summary; lower = better.
**Coverage-at-risk** — for a target `r_max` (e.g. 1 % selective error),
pick the largest `τ` with `e(τ) ≤ r_max`.

## When to use

- **High-stakes automation** — medical triage, credit decisions, safety
  systems where a "human review" queue exists.
- **Any classifier + any confidence score** — no retraining needed.
- **Cost-sensitive decisions** — abstention cost `< ` misprediction cost.

## Combining with other UQ

- **MSP / softmax entropy** — fastest baseline.
- **Conformal set size** — |set(x)| = 1 acts; else abstain (see
  `conformal-classification`).
- **MC dropout / deep ensemble variance** — abstain when disagreement high.
- **Evidential vacuity** — abstain when `K/S > threshold`.

## Files

- `python/selective_prediction.py` — from-scratch risk-coverage curve,
  AURC, coverage-at-risk on a synthetic 4-class softmax classifier
  (n = 1000, 15 % hard examples). Demo shows selective accuracy climbs
  from 0.660 at full coverage to 0.825 at 20-40 % coverage; AURC = 0.20;
  coverage under 5 % selective error target = 6.4 %.
- `r/selective_prediction.R` — pure R (any threshold on a score column)
  or `reticulate` + Python selective libraries.

## Assumptions & caveats

- **Coverage-vs-risk is a trade-off** — reject too much and you lose
  utility; reject too little and error stays high.
- **Confidence-score choice matters** — MSP is a decent baseline;
  ensemble / conformal / evidential scores typically outperform.
- **Rejection rate calibration** — the empirical `c(τ)` on your
  calibration set may not match deployment if the input distribution
  shifts (see `covariate-shift-adaptation`).
- **Fairness** — abstention rates can differ across subgroups; measure
  before deploying.
- **Cost matrix** — the optimal threshold depends on the cost of a
  wrong prediction vs the cost of abstention.

## Related in this repo

- `conformal-classification` — |set| = 1 acts as an abstention flag with
  a coverage guarantee.
- `ood-detection` — an OOD score can gate abstention on top of confidence.
- `mc-dropout`, `deep-ensembles`, `evidential-deep-learning` — richer
  confidence signals.
- `calibration-scaling` — a well-calibrated `g(x)` makes the threshold
  transferable across data.

## Run

```
python techniques/selective-prediction/python/selective_prediction.py
Rscript techniques/selective-prediction/r/selective_prediction.R
```

**Refs:** Chow, C.K. "An optimum character recognition system using decision functions." *IRE Transactions on Electronic Computers*, 1957; El-Yaniv, R. & Wiener, Y. "On the foundations of noise-free selective classification." *JMLR*, 2010; Geifman, Y. & El-Yaniv, R. "Selective classification for deep neural networks." *NeurIPS*, 2017.

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
