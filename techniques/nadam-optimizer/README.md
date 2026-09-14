# Nadam — Nesterov Adam (Reference §47.354)

Dozat (2016). Blend of Adam and Nesterov momentum. Uses a
LOOK-AHEAD version of the first-moment estimate:

```
m ← β₁ m + (1 − β₁) g
v ← β₂ v + (1 − β₂) g²
m̂, v̂ ← bias-correct(m, v)
m_nesterov ← β₁ m̂ + (1 − β₁) g / (1 − β₁ᵗ)
θ ← θ − η · m_nesterov / (√v̂ + ε)
```

Same memory as Adam. Slightly faster convergence in practice
because the Nesterov look-ahead damps late-stage oscillations.

## Files

- `python/nadam_optimizer.py` — Rosenbrock d=10 at
  `x₀ = −1, lr = 0.02`, 4 000 steps. Nadam reaches `f = 0.14`
  and `x̄ = 0.88`; Adam reaches `f = 0.21` and `x̄ = 0.85`.
- `r/nadam_optimizer.R` — `keras::optimizer_nadam` (R);
  `torch.optim.NAdam`, `optax.nadam`, from-scratch (Python).

## When to use

- **Drop-in Adam replacement** when the LR schedule allows it.
- **Recurrent / attention nets** — Dozat's original evaluation
  showed clearer wins on RNN-LM than on CNN.
- **When Adam plateaus late in training** — Nesterov's
  anticipatory step can shave off the last percentage points.

## When NOT to use

- **When AdamW's decoupled weight decay is critical** — use
  AdamW; Nadam variants (Nadam+W) exist but are less battle-
  tested.
- **When you need Adafactor's memory savings** — Nadam has
  same memory as Adam.
- **Very short training runs** — the gain over Adam is
  marginal until convergence effects matter.

## Assumptions & caveats

- **Same hyperparameters as Adam** — no re-tuning needed in
  most cases; β₁ = 0.9, β₂ = 0.999.
- **`(1 − β₁ᵗ)` warmup** — early steps are slightly larger
  than Adam's; usually not an issue.
- **Bias-correction ordering** — the Nesterov term uses the
  bias-corrected `m̂` and the raw `g / (1 − β₁ᵗ)`.
- **Not the same as Adam + Nesterov momentum** — full
  derivation reweights the update, not just the momentum.

## Related in this repo

- `adam-optimizer`, `adamw-decoupled-weight-decay`,
  `rmsprop-optimizer`, `adagrad`, `lion-optimizer` —
  first-order alternatives.
- `nesterov-accelerated-gradient` — the momentum variant
  Nadam adapts.
- `rectified-adam-radam`, `lookahead-optimizer` — modern
  Adam refinements.
- `lr-schedules`, `one-cycle-super-convergence` — LR
  scheduling cousins.

## Run

```
python techniques/nadam-optimizer/python/nadam_optimizer.py
Rscript techniques/nadam-optimizer/r/nadam_optimizer.R
```

**Refs:** Dozat, T. "Incorporating Nesterov momentum into Adam." *ICLR Workshop*, 2016.

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
