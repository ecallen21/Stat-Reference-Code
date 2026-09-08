# Tree of Thoughts (ToT) (Reference §47.92)

Yao et al (2023). Generalises chain-of-thought and self-consistency:

    1. GENERATE b candidate "thoughts" (intermediate reasoning steps).
    2. EVALUATE each with an LLM-scoring heuristic.
    3. SEARCH the tree by BFS / DFS with pruning.

Beats vanilla and self-consistency CoT on Game-of-24, Creative
Writing, Crosswords — problems where SEARCH matters, not just
sampling.

## Files

- `python/tree_of_thoughts.py` — Game-of-24 symbolic solver
  comparing (a) random one-shot CoT chains vs (b) beam-search ToT
  with proximity-to-24 heuristic. Demo (60 solvable puzzles):
  - Random CoT budget=100 chains  → 0.35 acc
  - **ToT BFS beam=32              → 0.78 acc**
  - ToT BFS beam=8                → 0.47 acc (vs 20-chain CoT 0.05).
- `r/tree_of_thoughts.R` — LLM tooling in Python; `langchain`,
  `tree-of-thoughts`, `guidance`.

## When to use

- **Search-hard reasoning** — puzzles, symbolic proofs, planning.
- **Long-horizon LLM planning** — decompose into rated sub-goals.
- **Games with pruning heuristics** — LLM-as-evaluator style.
- **When self-consistency plateaus** — extra compute better spent
  on structured search.

## When NOT to use

- **Simple factual QA** — CoT / retrieval is cheaper.
- **Ill-defined evaluator** — the search is only as good as the
  scoring function; noisy scores → thrashing.
- **Latency-critical** — deeper trees explode compute quadratically.

## Assumptions & caveats

- **Branching factor b** and depth trade off against compute.
- **Beam width** critical — too narrow prunes solutions; too wide
  = brute search.
- **Evaluator alignment** — LLM-as-judge biases carry into search.
- **Backtracking / DFS** with value estimation extends the framework.

## Related in this repo

- `chain-of-thought-reasoning`, `self-consistency-prompting`,
  `in-context-learning`, `retrieval-augmented-generation`,
  `dpo-direct-preference-optimization`, `rlhf-preferences` —
  LLM reasoning / alignment stack.
- `monte-carlo-tree-search`, `mdp-value-iteration`,
  `hierarchical-rl-options` — search / planning toolkit.
- `beam-search` (via `text-generation-decoding`),
  `speculative-decoding` — decoding cousins.

## Run

```
python techniques/tree-of-thoughts/python/tree_of_thoughts.py
Rscript techniques/tree-of-thoughts/r/tree_of_thoughts.R
```

**Refs:** Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T.L., Cao, Y. & Narasimhan, K. "Tree of Thoughts: Deliberate problem solving with large language models." *NeurIPS*, 2023.

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
