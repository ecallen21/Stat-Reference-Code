# Cross-Entropy / Log-Loss (Reference §34.6)

The **training loss** of every softmax / logistic classifier.

## Formulas

```
Binary:      L(y, p̂) = − [ y log p̂ + (1 − y) log(1 − p̂) ]
Categorical: L(y, p̂) = − Σ_k y_k log p̂_k
```

## Identities

- **Cross-entropy = entropy + KL**: `H(y, p̂) = H(y) + KL(y ‖ p̂)` when
  `y` is a distribution.
- **Softmax gradient**: `∂L/∂z = p̂ − y_onehot` (cancels the
  numerically-nasty softmax derivative).
- **MLE = argmin cross-entropy** for the corresponding likelihood.
- **Proper scoring rule** — uniquely minimised at the true probability.

## When to use

- **Any classifier / probabilistic prediction** — the default loss.
- **Model calibration** evaluation via log-loss (proper score).
- **Training discrete generative models** (softmax next-token
  prediction in LMs).

## When NOT to use

- **Zero-inflated / long-tail** classes — heavy imbalance requires
  focal loss or class-balanced variants.
- **Ordinal labels** — proportional-odds log-loss preserves ordering.
- **Regression** — use MSE / Huber / quantile loss.

## Files

- `python/cross_entropy_log_loss.py` —
  1. MLE minimum of log-loss demo: shifted probabilities strictly
     worse.
  2. **Softmax gradient identity**: analytic `(p̂ − y)` matches
     numerical to `2 × 10⁻¹¹`.
  3. **Proper-scoring rule check**: minimise `E_y[log loss(y, q)]`
     over `q`; recovered `q̂ = 0.70` = true probability.
- `r/cross_entropy_log_loss.R` — `MLmetrics::LogLoss`,
  `yardstick::mn_log_loss` (R); `sklearn.metrics.log_loss`,
  `torch.nn.CrossEntropyLoss` (Python).

## Assumptions & caveats

- **Numerical stability** — clip probabilities to `[ε, 1 − ε]` (or
  use log-softmax + NLL directly).
- **Class weighting** — for imbalanced classes.
- **Not calibration-aware** — a model with high log-loss but strong
  ranking (AUROC) can still discriminate; calibrate to reduce log-loss.
- **Multi-label** — use element-wise BCE, not categorical CE.

## Related in this repo

- `shannon-entropy`, `kl-divergence`, `f-divergences` — sibling
  info-theoretic quantities.
- `logistic-regression`, `multinomial-logit` — models trained with CE.
- `calibration-scaling`, `calibration-parity` — improve deployment log-loss.
- `focal-loss` (adjacent) — variant for imbalanced classes.

## Run

```
python techniques/cross-entropy-log-loss/python/cross_entropy_log_loss.py
Rscript techniques/cross-entropy-log-loss/r/cross_entropy_log_loss.R
```

**Refs:** Shannon, C.E. "A mathematical theory of communication." *Bell System Technical Journal*, 1948; Good, I.J. "Rational decisions." *JRSS-B*, 1952 (proper scoring rules).

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
