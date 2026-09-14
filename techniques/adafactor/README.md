# Adafactor (Reference §47.355)

Shazeer & Stern (2018). Sub-linear-memory adaptive optimiser
that FACTORISES the second-moment tensor into an outer product
of row/column statistics for each weight MATRIX:

```
U = g² + ε₁
R ← β₂ R + (1 − β₂) sum_j U[i, j]
C ← β₂ C + (1 − β₂) sum_i U[i, j]
v_est[i, j] ≈ R[i] C[j] / sum(R)
θ ← θ − η · g / (√v_est + ε₂)
```

For a `(rows × cols)` weight matrix, state drops from
`rows · cols` (Adam's `m + v`) to just `rows + cols`. Used
to train T5, PaLM, and other very large transformers where
optimiser state would otherwise blow past GPU memory.

## Files

- `python/adafactor.py` — 40×30 weight matrix regression
  problem (n=200). Both Adafactor and AdamW recover the
  true `‖W‖ = 10.27` to 3 decimals (MSEs 0.061 vs 0.036).
  Memory: Adafactor 10 160 B (W + R + C) vs AdamW 28 800 B
  (W + m + v) → **64.7 % savings**.
- `r/adafactor.R` — reticulate to `transformers.Adafactor` or
  `optax.adafactor` (R); `transformers.optimization.Adafactor`,
  `keras.optimizers.Adafactor`, from-scratch (Python).

## When to use

- **Very large transformers** — T5, PaLM, Whisper, mT5.
- **GPU memory-constrained fine-tuning** — Adafactor state
  is a fraction of AdamW's for matrix weights.
- **When you can tune LR schedules carefully** — Adafactor
  is stable with the recommended `relative_step` + external
  LR schedule.

## When NOT to use

- **Small models** — the memory savings don't matter.
- **1D parameters (LayerNorm, biases)** — Adafactor falls
  back to a scalar second moment; no factorisation gain.
- **Precise AdamW-tuned recipes** — the two aren't identical;
  hyperparameters must be re-tuned.

## Assumptions & caveats

- **RMS clipping / relative step** — the original paper
  introduces update-magnitude clipping to stabilise training;
  many implementations enable it by default.
- **No first-moment** — Adafactor is often used
  momentum-free (`beta1 = 0`); using momentum reintroduces
  the full `m` tensor and defeats the memory savings.
- **Learning rate parameterisation** — Adafactor uses a
  relative-step LR that scales with parameter RMS; treat it
  differently from Adam's absolute LR.
- **Only applies to 2D+ tensors** — biases and scalars keep
  full-precision second moments.

## Related in this repo

- `adamw-decoupled-weight-decay`, `adam-optimizer`,
  `lion-optimizer`, `nadam-optimizer` — dense-state
  neighbours.
- `zero-redundancy-optimizer`, `fsdp-fully-sharded-data-parallel`
  — distributed alternative to reducing optimiser memory.
- `mixed-precision-training`, `gradient-checkpointing` —
  orthogonal memory-saving techniques.
- `bitsandbytes-int8-llm`, `kv-cache-quantization` — model-
  weight quantisation cousins.

## Run

```
python techniques/adafactor/python/adafactor.py
Rscript techniques/adafactor/r/adafactor.R
```

**Refs:** Shazeer, N. and Stern, M. "Adafactor: adaptive learning rates with sublinear memory cost." In *ICML*, 2018.

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
