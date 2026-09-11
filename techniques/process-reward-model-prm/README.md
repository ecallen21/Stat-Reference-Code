# Process Reward Model — PRM (Reference §47.195)

Lightman et al. (2023, OpenAI PRM800K); Wang et al. (2024,
Math-Shepherd). Reward models for reasoning come in two flavours:

- **Outcome-Reward-Model (ORM)**: score only the final answer.
- **Process-Reward-Model (PRM)**: score EACH intermediate step.

PRMs identify where the reasoning goes wrong even if the final
answer is right by luck. Combined with:

- PRM-weighted best-of-N sampling: pick the sample with highest
  MIN or MEAN step reward.
- PRM-guided beam / MCTS search over reasoning.

## Files

- `python/process_reward_model_prm.py` — toy 8 + 5 − 3 = ? MATH
  problem with 6 sampled chains-of-thought:
  - Chain B (**lucky** wrong steps, right answer): ORM = 1.0, PRM
    = **0.30** — PRM correctly downgrades the lucky guess.
  - Chain C (right first step, wrong answer): ORM = 0.0, PRM =
    **0.70** — PRM credits partial reasoning.
  - Best-of-N tiebreak: PRM prefers chain with correct steps.
- `r/process_reward_model_prm.R` — no R port; recommends
  `openai/prm800k`, Math-Shepherd, `trl.RewardTrainer`.

## When to use

- **Multi-step reasoning** (math, code, planning) where step
  correctness matters.
- **RLHF / DPO with reasoning tasks** — PRM gives denser reward
  signal per-step vs sparse per-answer.
- **Selecting from many samples** — best-of-N with PRM beats
  ORM on MATH (Lightman 2023: 78 % vs 72 %).

## When NOT to use

- **Tasks with only one 'step'** — chat responses, translations
  — PRM adds no signal.
- **When step labels are prohibitively expensive** — PRM800K
  needed ~800k human annotations.
- **Ambiguous decompositions** — some tasks have many valid
  reasoning paths.

## Assumptions & caveats

- **Step boundary detection** — chain-of-thought steps need
  reliable segmentation ("Step 1:", newlines, etc.).
- **Label supervision** — human-labeled (PRM800K) or auto-labeled
  by rollout success (Math-Shepherd).
- **Aggregation** — min-of-step-rewards is common (worst-case);
  mean also works.
- **Reward hacking** — model can pad reasoning with trivial
  "correct" steps to inflate PRM score.

## Related in this repo

- `rlhf-preferences`, `dpo-direct-preference-optimization`,
  `constitutional-ai` — preference-alignment neighbours.
- `best-of-n-sampling` — the natural downstream selection method.
- `tree-of-thoughts-reasoning`,
  `chain-of-thought-reasoning` — reasoning-time cousins that
  benefit from PRM guidance.

## Run

```
python techniques/process-reward-model-prm/python/process_reward_model_prm.py
Rscript techniques/process-reward-model-prm/r/process_reward_model_prm.R
```

**Refs:** Lightman, H. et al. "Let's verify step by step." *ICLR*, 2024; Wang, P. et al. "Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations." *ACL*, 2024.

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
