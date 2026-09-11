# Verifier-Guided Search (Reference §47.205)

Uesato et al. (2022); Lightman (2023); Wang (2024, Math-Shepherd).
Combine LLM generation with a learned **verifier** (reward / value
model) to prune bad reasoning paths during beam search or MCTS:

    for step t:
        candidates = generate_next_step(context)
        scores = verifier(context, candidates)
        keep top-b beams by score
    Backtrack from best final beam.

Substantially better sample efficiency than plain best-of-N when
compute is limited.

## Files

- `python/verifier_guided_search.py` — toy arithmetic-target task
  (reach `target` in `depth` steps using +2/+3/·2/−1). 5 tasks:
  - **Greedy-N=20: 1/5 solved**.
  - **Verifier-beam=3: 4/5 solved** — pruning by proximity-to-
    target reaches most puzzles.
- `r/verifier_guided_search.R` — no R port; recommends
  `llm-reasoners`, `princeton-nlp/tree-of-thought-llm`,
  `openai/prm800k`.

## When to use

- **Reasoning tasks** with a per-step verifiable signal (math,
  code, logic).
- **When a strong verifier is cheaper than more generation** —
  test-time compute pareto.
- **Combined with PRM** for reasoning; ORM works for outcome-
  only.

## When NOT to use

- **When verifier is noisy** — verifier-beam can prune the right
  path.
- **When the answer space has no clear intermediate score**
  (creative writing, dialogue).
- **Very short tasks** — beam width 1 (greedy) is enough.

## Assumptions & caveats

- **Verifier calibration** — train on realistic partial
  reasoning, not just final answers.
- **Beam width** trades quality vs. cost; 3-16 typical.
- **Interaction with sampling temperature** — higher T ⇒ more
  diverse candidates for the verifier to filter.
- **Combined with MCTS** (AlphaMath, TS-LLM 2024) — search tree
  guided by value estimate.

## Related in this repo

- `process-reward-model-prm` — the verifier used in reasoning
  search.
- `tree-of-thoughts-reasoning`, `beam-search-decoding` —
  search-side neighbours.
- `best-of-n-sampling` — simpler alternative at high N.
- `monte-carlo-tree-search` — game-playing search cousin.
- `reflexion-self-critique` — related self-improvement loop.

## Run

```
python techniques/verifier-guided-search/python/verifier_guided_search.py
Rscript techniques/verifier-guided-search/r/verifier_guided_search.R
```

**Refs:** Uesato, J. et al. "Solving math word problems with process- and outcome-based feedback." *arXiv:2211.14275*, 2022; Lightman, H. et al. "Let's verify step by step." *ICLR*, 2024; Wang, P. et al. "Math-Shepherd." *ACL*, 2024.

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
