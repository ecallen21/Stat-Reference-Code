# Prefix / Prompt Tuning (Reference §47.111)

Li & Liang (2021, prefix); Lester, Al-Rfou & Constant (2021, prompt).
FREEZE a pre-trained transformer; learn only a small tensor of
prompt vectors prepended to every layer's key/value cache (prefix)
or the input embeddings (prompt).

Trainable parameters typically 0.01–0.1 % of the base model, yet
approach full fine-tuning at large scale (Lester's headline result).

## Files

- `python/prefix_prompt_tuning.py` — analogue with a FROZEN
  random projection (proxy pretrained encoder) + trainable
  soft-prompt vector prepended to the encoded features. Demo
  (n=400, d=20, d_pretrained=64):
  - full fine-tune (65 params): acc 1.000
  - prompt tuning with just 2 / 4 / 8 / 16 prompt params:
    **all reach acc 1.000** on this separable task.
- `r/prefix_prompt_tuning.R` — no first-class R port; `peft`
  (Hugging Face), `transformers` in Python.

## When to use

- **Fine-tune massive LLMs** on modest data — parameter-efficient.
- **Serve many tasks** from a single base model + tiny prompts
  per task.
- **Federated / privacy-preserving fine-tuning** — share prompts
  only.
- **Prompt-only research** on frozen SOTA models.

## When NOT to use

- **Tasks requiring deep structural change** to the base model —
  full fine-tune or LoRA-adapter more expressive.
- **Very small base models** (< 1 B params) — Lester 2021 shows
  gap widens; prefer full FT.
- **Extremely OOD tasks** — soft prompts may not steer the model
  far enough.

## Assumptions & caveats

- **Optimisation is non-convex** — soft prompts prone to local
  optima; initialise from natural-language tokens or random.
- **Prompt length** trade-off — longer = more capacity, longer
  context = higher inference cost.
- **Adapter vs prefix vs prompt vs LoRA**: pick by budget +
  serving constraints.
- **Instruction tuning** distinct — full-data supervised FT on
  many tasks.

## Related in this repo

- `lora-peft`, `knowledge-distillation`, `transfer-learning`,
  `meta-learning-maml` — parameter-efficient / adaptation family.
- `in-context-learning`, `chain-of-thought-reasoning`,
  `self-consistency-prompting`, `tree-of-thoughts` — prompting
  cousins that need no parameter updates at all.
- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  alignment methods that DO fine-tune the base model.
- `transformer-encoder`, `transformer-decoder` — model families
  soft prompts steer.

## Run

```
python techniques/prefix-prompt-tuning/python/prefix_prompt_tuning.py
Rscript techniques/prefix-prompt-tuning/r/prefix_prompt_tuning.R
```

**Refs:** Li, X.L. & Liang, P. "Prefix-tuning: Optimizing continuous prompts for generation." *ACL*, 2021; Lester, B., Al-Rfou, R. & Constant, N. "The power of scale for parameter-efficient prompt tuning." *EMNLP*, 2021.

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
