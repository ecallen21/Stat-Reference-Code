# LoRA -- Low-Rank Adaptation (Reference §47.24)

Hu et al. (2021, ICLR). Rather than fine-tuning all weights
`W ∈ ℝ^{d × k}`, add a low-rank update:

    W_new = W_frozen + (α / r) · B · A
    A ∈ ℝ^{r × k},  B ∈ ℝ^{d × r},  r ≪ min(d, k)

Only `A, B` are trainable — `r(d + k)` parameters vs `dk`. Merges to
a single `W_new` at inference (zero latency overhead).

## Family

- **QLoRA** (Dettmers 2023) — 4-bit base + LoRA adapters.
- **DoRA** (Liu 2024) — decompose direction and magnitude.
- **AdaLoRA** — adaptive rank per layer.
- **IA³** — even fewer parameters via gating.
- **Prefix / prompt tuning** — earlier PEFT precursors.

## Files

- `python/lora_peft.py` — toy LoRA vs full fine-tune on a small
  linear layer from scratch (manual gradients through `B · A`).
  Demo (d=40, k=30, r ∈ {2, 4, 8}): LoRA trains 140–560 params (vs
  1200 for full FT) at increasing MSE — a full-rank synthetic target
  makes the tradeoff visible. In real LLMs the update is empirically
  low-rank, so LoRA usually matches full FT.
- `r/lora_peft.R` — no R implementations; describes HuggingFace
  `peft`, `trl`, bitsandbytes, unsloth, lit-gpt, axolotl.

## When to use

- **Fine-tuning large LMs on a modest GPU** — 7B–70B parameter
  models with 24 GB (or 4-bit QLoRA on smaller GPUs).
- **Multiple task adapters** — swap in / out adapters at inference
  without duplicating base weights.
- **Rapid iteration** — checkpoint sizes MB not GB.

## When NOT to use

- **Small model, plenty of GPU** — full FT is simpler and possibly
  better.
- **Task needs adjustments to embeddings / early layers not touched
  by LoRA** — apply LoRA to more modules or use DoRA / AdaLoRA.
- **You need to update the base weights** — LoRA is additive, not a
  replacement.

## Assumptions & caveats

- **Rank `r` and scaling `α`** — usually `r ∈ {4, 8, 16, 32, 64}`
  and `α = 2r`; sweep.
- **Which layers to adapt** — typically Q, K, V, O in attention +
  optionally MLP; more layers → more params.
- **Zero-init `B`** — required so the initial forward pass reproduces
  the base model exactly.
- **Merging** — for inference collapse `W_new = W + BA/r` into one
  matrix (no latency), but you lose the swappable adapter.

## Related in this repo

- `transfer-learning`, `knowledge-distillation`,
  `meta-learning-maml` — model-reuse cousins.
- `transformer-decoder`, `attention-mechanism`, `retrieval-augmented-generation`
  — LM machinery.
- `rlhf-preferences`, `dpo-direct-preference-optimization` — align
  fine-tuned models.

## Run

```
python techniques/lora-peft/python/lora_peft.py
Rscript techniques/lora-peft/r/lora_peft.R
```

**Refs:** Hu, E.J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022; Dettmers, T. et al. "QLoRA: efficient finetuning of quantized LLMs." *NeurIPS*, 2023; Liu, S.-Y. et al. "DoRA: Weight-Decomposed Low-Rank Adaptation." *ICML*, 2024.

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
