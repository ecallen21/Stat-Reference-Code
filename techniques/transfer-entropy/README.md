# Transfer Entropy (Reference §34.11)

Schreiber (2000). Quantifies **directed information flow** from
process `X` to `Y`.

## Formula

```
TE(X → Y)  =  H(Y_{t+1} | Y_t^{(k)})  −  H(Y_{t+1} | Y_t^{(k)}, X_t^{(l)}).
```

Non-zero TE means `X`'s history reduces uncertainty about `Y_{t+1}`
**beyond** `Y`'s own history — the non-linear cousin of Granger
causality.

## When to use

- **Directed connectivity** in neural / financial / physiological
  time series.
- **Extension of Granger causality** without a linear assumption.
- **Complex-systems flow** analysis.

## When NOT to use

- **Small n** — TE is data-hungry; needs `n ≥ 1000` per series for
  reliable estimates.
- **Continuous-valued signals** with fine dynamics — kernel /
  KSG-style estimators required.
- **Confounded systems** — TE detects information flow, not causation
  in the Pearl sense.

## Files

- `python/transfer_entropy.py` — from-scratch discretised (quartile-
  bin) TE with lag-1 history. Demo:
  - **X drives Y** (`Y_{t+1} = 0.5 Y_t + 0.7 X_t + noise`):
    `TE(X→Y) = 0.50`; `TE(Y→X) = 0.009` (correctly directional).
  - **Independent**: both TEs ≈ 0.01 (baseline discretisation noise).
- `r/transfer_entropy.R` — `RTransferEntropy` (R); `IDTxl`, `PyIF`,
  JIDT (Python / Java).

## Assumptions & caveats

- **Stationarity** required for the joint distributions.
- **Discretisation choice** — quartile bins are a compromise; KSG
  estimators avoid the choice at higher compute cost.
- **Lag order** — the demo uses lag 1; longer histories add DoF.
- **Statistical significance** — permutation test on the source series.
- **Effective vs full TE** — subtract chance-level TE for calibrated
  values.
- **Not causation** — TE detects predictive information flow, subject
  to unobserved confounders.

## Related in this repo

- `mutual-information`, `conditional-mutual-info` — symmetric siblings.
- `shannon-entropy` — the building block.
- `granger-causality` (adjacent) — linear cousin.
- `structural-equation-modeling`, `bayesian-networks` — causal
  alternatives.

## Run

```
python techniques/transfer-entropy/python/transfer_entropy.py
Rscript techniques/transfer-entropy/r/transfer_entropy.R
```

**Refs:** Schreiber, T. "Measuring information transfer." *Physical Review Letters*, 2000; Lizier, J.T. et al. "Local measures of information storage in complex distributed computation." *Information Sciences*, 2011.

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
