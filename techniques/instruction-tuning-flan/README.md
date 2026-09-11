# Instruction Tuning / FLAN (Reference §47.182)

Wei et al. (2022, ICLR). Take a pretrained LM and fine-tune it on
**many NLP tasks reformulated as instruction-following**:

    "Translate this sentence to French: {source}"    → {target}
    "Summarize the following passage: {passage}"     → {summary}
    "Is the sentiment positive, negative, or neutral? {text}" → {label}

At inference the model generalises to **new instruction types**
(zero-shot) it never saw at training. Foundation of ChatGPT-style
assistants; multi-task training + natural-language task
descriptions is the key.

## Files

- `python/instruction_tuning_flan.py` — toy bag-of-embeddings LM
  fine-tuned on two "classify the …" tasks (sentiment, topic),
  then evaluated on a held-out language-identification task.
  - Real FLAN-T5 / T0 / Tulu-2 do far better than this
    illustrative bag-of-tokens simulation; the point is to show
    the instruction-formatting protocol (prompt template →
    natural-language target).
- `r/instruction_tuning_flan.R` — no R port; recommends
  `transformers.Trainer`, `trl.SFTTrainer`, FLAN-T5 / T0 / Tulu
  checkpoints on HF Hub.

## When to use

- **Turning a base LM into an instruction-following assistant** —
  the standard first step after pretraining.
- **When you have (or can generate) many diverse task examples**
  in instruction format.
- **Domain-specific assistants** — instruction-tune on a curated
  corpus of your domain's tasks.

## When NOT to use

- **When the task is a single, static classification** — plain
  supervised fine-tuning is simpler.
- **When you need alignment with human preferences** — instruction
  tuning is upstream of RLHF / DPO / Constitutional AI.
- **Very small models** — instruction tuning shines from ~1 B
  params up.

## Assumptions & caveats

- **Task diversity** matters most — FLAN uses 60+ NLP tasks.
- **Prompt template stability** — hold-out generalises only if
  test-time prompts match training-time formatting.
- **Length bias** — instruction tuning can over-favour long,
  verbose outputs; adjust with length-normalised objectives.
- **Data quality** > quantity — Tulu-2, WizardLM findings show a
  small curated set beats a huge noisy one.

## Related in this repo

- `constitutional-ai` — RL-side alignment step after SFT.
- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  preference-alignment cousins.
- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning` — reasoning-time cousins that
  build on instruction-tuned base models.
- `lora-peft`, `prefix-prompt-tuning` — parameter-efficient
  fine-tuning options for the SFT step.

## Run

```
python techniques/instruction-tuning-flan/python/instruction_tuning_flan.py
Rscript techniques/instruction-tuning-flan/r/instruction_tuning_flan.R
```

**Refs:** Wei, J. et al. "Finetuned language models are zero-shot learners." *ICLR*, 2022; Chung, H. W. et al. "Scaling instruction-finetuned language models." *arXiv:2210.11416*, 2022; Sanh, V. et al. "Multitask prompted training enables zero-shot task generalization (T0)." *ICLR*, 2022.

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
