# Reflexion — Self-Critique with Verbal RL (Reference §47.201)

Shinn et al. (2023, NeurIPS). Instead of gradient RL, use
**self-generated verbal feedback** stored in an EPISODIC MEMORY:

    for trial t = 1..T:
        action = LM(prompt, memory)
        result = env(action)
        if success: break
        reflection = LM("what went wrong? how to fix?", action, result)
        memory.append(reflection)

Beats greedy CoT on HumanEval / HotpotQA / AlfWorld by 5-20 pts.

## Files

- `python/reflexion_self_critique.py` — toy Reflexion on 6
  arithmetic puzzles (find op such that a·b·c = target):
  - **Baseline (no memory): 4/6 solved**
  - **Reflexion (memory of failed actions): 5/6 solved** — extra
    puzzle recovered because memory prevents retrying the same
    wrong operation.
- `r/reflexion_self_critique.R` — no R port; recommends
  `noahshinn024/reflexion`, `langgraph` reflection agents.

## When to use

- **Tool-using agents** where feedback from the environment is
  rich (test failures, compiler errors).
- **When gradient-based RL is impractical** (closed-source LM,
  ephemeral compute).
- **Iterative refinement** tasks (code, planning, retrieval).

## When NOT to use

- **Single-step tasks** — no room for reflection.
- **Very short LM context** — memory can't accumulate meaningfully.
- **When rewards are dense enough for gradient RL** — RLHF /
  DPO is more sample-efficient.

## Assumptions & caveats

- **Memory format** — natural-language reflections; the LM must
  compress and prioritise them well.
- **Retry budget** — 3-5 trials typical; more risks getting
  stuck in a bad hypothesis.
- **Verifier quality** — Reflexion needs a reliable way to
  detect failure (test suite, env reward, LM critic).
- **Not a training method** — the base model is frozen; memory
  is per-episode.

## Related in this repo

- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning` — cousin reasoning-time methods.
- `react-reasoning-acting` — tool-use companion.
- `constitutional-ai`,
  `rlhf-preferences` — training-time alignment cousins.
- `process-reward-model-prm` — the reward-model side of
  reasoning-time signals.

## Run

```
python techniques/reflexion-self-critique/python/reflexion_self_critique.py
Rscript techniques/reflexion-self-critique/r/reflexion_self_critique.R
```

**Refs:** Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K. & Yao, S. "Reflexion: Language agents with verbal reinforcement learning." *NeurIPS*, 2023; Madaan, A. et al. "Self-Refine: Iterative refinement with self-feedback." *NeurIPS*, 2023.

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
