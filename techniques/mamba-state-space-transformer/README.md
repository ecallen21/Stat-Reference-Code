# Mamba / Structured State-Space Models (S4, S5) (Reference §47.22)

Gu, Goel & Ré (S4, 2022); Gu & Dao (Mamba, 2023). A transformer
alternative built from **linear time-invariant** (S4) or
**input-selective** (Mamba) state-space recurrences.

## Discrete SSM

    h_t = A · h_{t−1} + B · x_t
    y_t = C · h_t     (+ D · x_t)

- **S4**: `A, B, C, D` fixed across the sequence; efficient via
  HiPPO initialisation + Cauchy-kernel FFT.
- **Mamba**: `B` and `C` are **input-dependent** (selective); scan
  training with `O(L · d)` compute and `O(1)` state per position.

## Two views (equivalent when LTI)

- **Recurrent**: online, streaming, `O(1)` memory per step.
- **Convolutional**: parallel over `t`, uses an impulse-response
  kernel `k[t] = C · Aᵗ · B` (`+D` at t=0).

## Files

- `python/mamba_state_space_transformer.py` — compact fixed-A SSM
  (S4-style without HiPPO tricks) from scratch. Demo (n=8, L=40):
  recurrent vs convolutional forward passes agree to `<10⁻¹⁶`.
  L=500 remains stable because `|eig(A)| < 1`; kernel energy in
  the first 20 taps = 98.5 % (effective receptive field).
- `r/mamba_state_space_transformer.R` — no R implementations;
  describes `mamba-ssm`, `state-spaces`, `s5-pytorch`, HF
  `transformers` checkpoints.

## When to use

- **Long-context sequence modelling** — DNA, audio, video, code —
  where attention's `O(L²)` cost dominates.
- **Streaming inference** — recurrent form is `O(1)` per token.
- **Small state / edge deployment** — SSM state is much smaller than
  a KV cache.

## When NOT to use

- **Short-context in-context learning** — attention still wins on
  many text tasks up to ~8k tokens.
- **Extreme accuracy at all costs** — hybrid architectures (Mamba +
  attention blocks) often beat pure Mamba on language benchmarks.

## Assumptions & caveats

- **Stability** — `|eig(A)| < 1` (or continuous-time `Re(λ) < 0`)
  is required; HiPPO gives principled initialisation.
- **Kernel length** — for LTI SSMs the effective receptive field
  depends on the eigenvalue spectrum; verify with kernel decay.
- **Selective (Mamba) SSMs** are not LTI — the FFT trick does not
  apply; use the associative scan.
- **Training tricks** — bfloat16 accumulation, gated projections,
  and Gated MLP fusions materially change accuracy.

## Related in this repo

- `state-space-kalman`, `state-space-models` — classical linear /
  nonlinear SSMs.
- `transformer-encoder`, `transformer-decoder`, `attention-mechanism`
  — the attention-based counterpart.
- `recurrent-nn`, `lstm-gru` — recurrent forerunners.

## Run

```
python techniques/mamba-state-space-transformer/python/mamba_state_space_transformer.py
Rscript techniques/mamba-state-space-transformer/r/mamba_state_space_transformer.R
```

**Refs:** Gu, A., Goel, K. & Ré, C. "Efficiently modeling long sequences with structured state spaces." *ICLR*, 2022; Gu, A. & Dao, T. "Mamba: linear-time sequence modeling with selective state spaces." *arXiv:2312.00752*, 2023; Smith, J.T.H. et al. "Simplified state space layers for sequence modeling" (S5). *ICLR*, 2023.

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
