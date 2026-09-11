# ReAct — Reasoning and Acting (Reference §47.202)

Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao (2023, ICLR).
Interleaves **reasoning** traces with **actions** in a tool-using
LM:

    Thought: "I need to look up the founding year..."
    Action: search[SpaceX founded 2002]
    Observation: SpaceX was founded in 2002...
    Thought: "Confirmed. So the answer is..."
    Action: finish[2002]

Combines the strengths of chain-of-thought (structured reasoning)
and tool use (external knowledge + actions on the world).

## Files

- `python/react_reasoning_acting.py` — toy ReAct agent with
  search + calculator tools on 3 questions ("When was SpaceX
  founded?", "What is 7 · 6 + 3?", etc.):
  - **CoT-only baseline: 0/3** (cannot look up facts or compute)
  - **ReAct: 3/3** (search finds year, calc computes 45).
- `r/react_reasoning_acting.R` — no R port; recommends
  `langchain.ReActAgent`, `llama-index.ReActAgent`, `dspy` ReAct.

## When to use

- **Tool-augmented LLMs** — the standard prompting pattern.
- **Multi-step queries** requiring both retrieval and reasoning.
- **When response quality matters more than latency** — ReAct
  adds tool round-trips.

## When NOT to use

- **Pure reasoning tasks** with no external state — plain CoT is
  simpler.
- **Ultra-low-latency serving** — tool calls add hundreds of ms.
- **Small LMs** — need a competent base to route between tools
  correctly.

## Assumptions & caveats

- **Prompt format** — the Thought/Action/Observation triples must
  be consistent; parsing errors are common.
- **Tool schema clarity** — each tool needs a crisp description.
- **Trajectory length** — cap steps to prevent infinite loops.
- **Combined with Reflexion** — reflections steer future ReAct
  trajectories.

## Related in this repo

- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning` — reasoning-time cousins.
- `toolformer-tool-use`, `function-calling-openai` — tool-
  learning cousins.
- `reflexion-self-critique` — memory-augmented ReAct variant.
- `autogen-multi-agent`,
  `crewai-hierarchical-agents` — multi-agent frameworks that
  wrap ReAct-style agents.

## Run

```
python techniques/react-reasoning-acting/python/react_reasoning_acting.py
Rscript techniques/react-reasoning-acting/r/react_reasoning_acting.R
```

**Refs:** Yao, S. et al. "ReAct: Synergizing reasoning and acting in language models." *ICLR*, 2023.

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
