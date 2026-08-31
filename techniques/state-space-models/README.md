# State-Space Models: S4 / Mamba (Reference §27.x extra)

Modern deep-learning primitive that transforms a sequence via a **linear
dynamical system** whose parameters are learned by SGD. Competitive with
transformers at `O(T)` compute instead of `O(T²)` attention.

## SSM layer

```
h_t = A · h_{t-1} + B · x_t              hidden dynamics
y_t = C · h_t + D · x_t                  output
```

Two equivalent computations:

- **Recurrent (scan) form** — sequential; `O(T · d_h²)`.
- **Convolutional form** — `y_t = Σ_k K[k] · x_{t−k}`, where `K[k] = C · A^k · B` is the SSM kernel. Computed via FFT in `O(T log T · d_h)`.

## S4 (Gu 2022)

- **HiPPO initialisation** of `A` (Legendre memory) — gives long-range memory out of the box.
- **Diagonal reparameterisation** — S4D (Gu 2022) makes `A` diagonal in complex domain; scalable and stable.
- Beats transformers on the Long Range Arena benchmark at 10–100× the compute efficiency.

## Mamba (Gu-Dao 2023)

- **Selective SSM**: `B`, `C`, `Δ` are **input-dependent** — the model chooses what to remember. Bridges the gap between transformers (input-conditioned attention) and classical SSMs (fixed dynamics).
- Linear-time training + inference.
- Competitive with Transformer++ on language modelling, DNA, audio.

## Family

| Model | Key idea |
|---|---|
| **S4** (Gu 2022) | HiPPO + diagonal + FFT convolution |
| **S5** (Smith 2023) | Simplified diagonal S4 |
| **Mamba** (Gu-Dao 2023) | Selective (input-dependent) SSM |
| **Mamba-2** (Dao-Gu 2024) | Structured state-space duality with attention |
| **Griffin / RecurrentGemma** (Botev 2024) | Hybrid SSM + local attention |
| **RWKV** (Peng 2023) | Linear-attention alternative with recurrent form |

## When to use

- **Very long context** — genomics, DNA, audio, long-document language modelling.
- **Real-time streaming inference** — SSMs run in constant memory per step.
- **Compute-bounded LLM inference** — linear rather than quadratic.
- **Time series / control** — the dynamical-system interpretation is natural.
- **NOT** as a strict transformer replacement yet — many benchmarks still favour attention for symbolic reasoning tasks; hybrid models are the current default.

## Files

- `python/state_space_models.py` — from-scratch SSM scan + convolutional-form kernel + impulse response on a random stable system. Sanity: scan-form and convolution-form outputs agree to 7e-16 (machine precision). Impulse response shows the decaying kernel; spectral radius 0.944 < 1 (stable).
- `r/state_space_models.R` — `KFAS::SSModel` for classical linear-Gaussian SSMs; `reticulate` + `mamba-ssm`, `s4d`, `state-spaces/s4`.

## Assumptions & caveats

- **Linear dynamics** — Mamba's selectivity adds input-dependent parameters but the per-step recurrence is still linear in `h`.
- **Stability** requires the spectral radius of `A` < 1; HiPPO init satisfies this.
- **FFT convolution** requires input length known in advance; use the recurrent form for streaming.
- **Diagonal A** loses expressiveness relative to dense `A` but is far faster and mathematically stable.
- **Training** — SSMs are more sensitive to init than transformers; use HiPPO or Mamba's paper-recommended init.
- **Selection interpretation** — Mamba's `Δ(x)` acts like a soft "gate" over how much to forget vs remember; interpretable as attention scores.

## Related in this repo

- `transformer-encoder`, `attention-mechanism` — the quadratic-attention neighbour.
- `recurrent-nn`, `lstm-gru` — the classical recurrent alternatives.
- `state-space-kalman` — the linear-Gaussian statistical counterpart.
- `functional-time-series` — related time-series representation-learning technique.

## Run

```
python techniques/state-space-models/python/state_space_models.py
Rscript techniques/state-space-models/r/state_space_models.R
```

**Refs:** Gu, A., Goel, K. & Ré, C. "Efficiently modeling long sequences with structured state spaces (S4)." *ICLR*, 2022; Gu, A. & Dao, T. "Mamba: linear-time sequence modeling with selective state spaces." *arXiv:2312.00752*, 2023; Dao, T. & Gu, A. "Transformers are SSMs: generalized models and efficient algorithms through structured state space duality." *ICML*, 2024.

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
