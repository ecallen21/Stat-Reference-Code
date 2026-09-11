# Constitutional AI — CAI (Reference §47.183)

Bai et al. (2022, Anthropic). Two-stage RLHF alternative that
reduces the need for human labels:

- **Stage 1 (SL): critique-and-revise loop.**
  For each response r:
  - critique = LM("critique r against principle P")
  - revised  = LM("revise r using the critique")
  Fine-tune the base model on (prompt, revised) pairs.

- **Stage 2 (RL): RLAIF (RL from AI Feedback).**
  Use an LM to score pairs of responses for harmfulness;
  train a preference model, then PPO the base model against it.

**"Constitution"** = set of natural-language principles (be helpful,
avoid harm, don't produce dangerous content, etc). CAI-trained
models are more harmless with less human data than pure RLHF.

## Files

- `python/constitutional_ai.py` — toy critique-and-revise loop on
  a bag-of-tokens harmfulness proxy. Three example prompts:
  - Before: preference-model score −4 / −2 / +1 (avg **−1.67**).
  - After critique-and-revise: **+2 / +2 / +1 (avg +1.67)** —
    harmful tokens rewritten to helpful ones.
  - Real CAI uses an LM to critique + revise; scoring model is a
    trained preference network.
- `r/constitutional_ai.R` — no R port; recommends `trl` /
  `trlx` / anthropic's public CAI paper.

## When to use

- **Aligning a base LLM** without extensive human red-teaming.
- **When principles can be stated in natural language** (safety,
  helpfulness, honesty).
- **Iterating on alignment** — new principles are added to the
  constitution and the loop re-runs cheaply.

## When NOT to use

- **When you have plentiful human preferences** and no compute
  budget for a critique LM — plain RLHF may be simpler.
- **Very small models** — the critique / revise LM must be
  competent to produce useful signal.
- **When the constitution can't be operationalised** — abstract
  ethics don't automatically become good critiques.

## Assumptions & caveats

- **Bootstrap problem** — the critique model itself must be
  aligned enough to produce useful revisions.
- **Principle interaction** — helpful + harmless can conflict;
  the constitution must be arbitrated.
- **Preference-model calibration** — biases in the RM propagate
  through PPO / DPO.
- **Not a safety guarantee** — CAI reduces obvious harm but
  doesn't remove adversarial jailbreaks.

## Related in this repo

- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  preference-alignment cousins.
- `instruction-tuning-flan` — upstream SFT step.
- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning` — reasoning-time cousins.
- `label-smoothing`, `randomized-smoothing` — sibling
  regularisation ideas (different domain).

## Run

```
python techniques/constitutional-ai/python/constitutional_ai.py
Rscript techniques/constitutional-ai/r/constitutional_ai.R
```

**Refs:** Bai, Y. et al. "Constitutional AI: Harmlessness from AI feedback." *arXiv:2212.08073*, 2022; Ouyang, L. et al. "Training language models to follow instructions with human feedback (InstructGPT)." *NeurIPS*, 2022.

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
