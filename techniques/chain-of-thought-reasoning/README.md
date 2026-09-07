# Chain-of-Thought + Self-Consistency (Reference §47.26)

Wei et al. (2022, CoT); Wang et al. (2022, self-consistency);
Kojima et al. (2022, zero-shot CoT).

## Chain-of-thought

Prompt the LM to write out **intermediate reasoning steps** ("Let's
think step by step") before the final answer. Dramatically boosts
multi-step reasoning accuracy for large models (emergent).

## Self-consistency

Sample `K` CoT trajectories (temperature > 0), extract final
answers, and **majority vote**. Amplifies accuracy over single-shot
CoT.

## Files

- `python/chain_of_thought_reasoning.py` — Monte-Carlo simulation
  of the majority-vote accuracy curve over K, treating each CoT
  sample as an iid Bernoulli. Demo (baseline p=0.30, single-CoT
  p=0.55): K=1 → 0.549, K=3 → 0.573, K=5 → 0.596, K=9 → 0.626,
  K=21 → 0.690, K=41 → 0.737. Diminishing returns as K grows.
- `r/chain_of_thought_reasoning.R` — `ellmer`, `chattr`,
  `gptstudio` (R); openai / anthropic / google-generativeai APIs +
  langchain / dspy / lm-eval-harness (Python).

## When to use

- **Multi-step reasoning tasks** — arithmetic, symbolic, code,
  common-sense chains.
- **Emergent-scale models** (>~100B) — CoT rarely helps small
  models.
- **Compute-for-accuracy budget** — self-consistency trades API
  calls for accuracy.

## When NOT to use

- **Single-token / classification tasks** — CoT overhead is
  wasteful.
- **Small models** — CoT sometimes HURTS accuracy in <10 B models
  (Wei et al. Fig 5).
- **Correctness-critical latency-tight applications** — the extra
  reasoning tokens add latency and cost.

## Assumptions & caveats

- **Prompt design** — the exemplar CoT reasoning traces matter;
  errors propagate.
- **Answer extraction** — needs a robust final-answer parser (regex
  / structured output).
- **Bias amplification** — majority vote can lock in a
  systematically wrong reasoning pattern.
- **Latent-thought / process reward models** (Lightman 2023) —
  supervise reasoning steps directly for further gains.

## Related in this repo

- `in-context-learning`, `retrieval-augmented-generation` —
  companion LM techniques.
- `text-generation-decoding`, `transformer-decoder`,
  `attention-mechanism` — LM machinery.
- `rlhf-preferences`, `dpo-direct-preference-optimization` — align
  a model to prefer good CoT reasoning.

## Run

```
python techniques/chain-of-thought-reasoning/python/chain_of_thought_reasoning.py
Rscript techniques/chain-of-thought-reasoning/r/chain_of_thought_reasoning.R
```

**Refs:** Wei, J. et al. "Chain-of-thought prompting elicits reasoning in large language models." *NeurIPS*, 2022; Wang, X. et al. "Self-consistency improves chain of thought reasoning in language models." *ICLR*, 2023; Kojima, T. et al. "Large language models are zero-shot reasoners." *NeurIPS*, 2022.

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
