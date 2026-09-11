# Tree of Thoughts — ToT (Reference §47.184)

Yao, Yu, Zhao, Shafran, Griffiths, Cao & Narasimhan (2023, NeurIPS).
Extends chain-of-thought by exploring **multiple reasoning paths as
a tree** and using explicit **search** (BFS / DFS) with value-based
pruning:

    1. Decompose problem → thought steps.
    2. At each node, generate k candidate next thoughts.
    3. Evaluate each candidate (via an LM 'value' call).
    4. Keep top-b beams; expand until solution or budget exhausted.

Beats single-chain CoT on planning / puzzle / creative tasks
(Game-of-24, Creative Writing, mini-crosswords).

## Files

- `python/tree_of_thoughts_reasoning.py` — from-scratch BFS ToT
  with a hand-written value function on Game-of-24 puzzles. Seven
  puzzles including easy (3, 8, 8, 8) and hard (5, 5, 5, 1):
  - **CoT-greedy: 2 / 7 solved**.
  - **ToT (beam = 8): 4 / 7 solved** — recovers puzzles greedy
    locks onto a bad first step.
- `r/tree_of_thoughts_reasoning.R` — no R port; recommends
  `princeton-nlp/tree-of-thought-llm`, `llm-reasoners`, `langgraph`.

## When to use

- **Multi-step reasoning** (arithmetic, planning, logic puzzles,
  code) where a wrong early step derails a chain.
- **When evaluation is cheaper than generation** — an LM value
  call is 1 forward pass, an expansion is 1 too.
- **Small-search-space** problems where BFS / beam fits the budget.

## When NOT to use

- **When single-chain CoT is enough** — ToT's compute overhead is
  significant.
- **Very long horizons** — tree grows exponentially without
  aggressive pruning.
- **Fuzzy / creative tasks with no clear value function** — the
  search reduces to random sampling.

## Assumptions & caveats

- **Value function quality** dominates ToT's advantage; a bad
  value collapses to random search.
- **Beam width** — 3-8 typical; larger = better recall, higher
  compute.
- **BFS vs DFS** — BFS is default; DFS with backtracking helps on
  deep-narrow trees.
- **Self-consistency** (Wang 2023) — cheaper alternative: sample
  many chains, vote.

## Related in this repo

- `chain-of-thought-reasoning` — the single-chain baseline ToT
  extends.
- `monte-carlo-tree-search` — game-playing search cousin (ToT
  is CoT-of-MCTS in some sense).
- `beam-search-decoding` — the token-level search primitive.
- `constitutional-ai`, `rlhf-preferences` — training-time
  alignment cousins.
- `instruction-tuning-flan` — upstream fine-tuning that unlocks
  ToT-style prompting.

## Run

```
python techniques/tree-of-thoughts-reasoning/python/tree_of_thoughts_reasoning.py
Rscript techniques/tree-of-thoughts-reasoning/r/tree_of_thoughts_reasoning.R
```

**Refs:** Yao, S. et al. "Tree of thoughts: Deliberate problem solving with large language models." *NeurIPS*, 2023; Wang, X. et al. "Self-consistency improves chain of thought reasoning in language models." *ICLR*, 2023.

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
