# Multi-Agent Debate (Reference §47.206)

Du, Li, Torralba, Tenenbaum & Mordatch (2023); Liang et al. (2023).
Multiple LM instances propose answers, then EACH sees the others'
answers and updates its own:

    round 0: A_0[i] = LM_i(question)
    round r: A_r[i] = LM_i(question, {A_{r-1}[j], j ≠ i})
    final: majority vote or last-round A_R[i]

Divergent thinking (Liang) explicitly prompts one agent as a
**critic** to challenge the majority. Substantially reduces
hallucinations on factual QA vs single-agent CoT.

## Files

- `python/multi_agent_debate.py` — toy 3-agent, 3-round debate on
  4 arithmetic questions with a 60 %-correct base agent:
  - **Single-agent accuracy: 55 %** per question.
  - **3-agent 3-round debate: 93 %** — +38 pp lift; convergence
    to truth via majority-vote update.
- `r/multi_agent_debate.R` — no R port; recommends
  `composable-models/llm-multiagent-debate`, `langgraph`,
  `autogen` GroupChatManager.

## When to use

- **Factual QA / arithmetic** — where an oracle-like majority
  vote works.
- **When you can afford N × single-agent cost**.
- **Combined with different-role prompts** (Liang) for divergent
  reasoning.

## When NOT to use

- **When single-agent CoT already saturates** (obvious tasks).
- **Divergent-answer tasks** (creative writing) — debate averages
  toward mediocrity.
- **When agents' errors are correlated** — same base model can
  agree on the wrong answer.

## Assumptions & caveats

- **Agent independence** — use different roles / prompts /
  temperatures to decorrelate.
- **Round budget** — 2-4 rounds typical; more can overfit to
  spurious consensus.
- **Majority vote vs last-round** — majority usually more
  robust.
- **Tie-breaking** — unresolved after debate → fall back to
  single-agent answer.

## Related in this repo

- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning`,
  `self-consistency-prompting` — reasoning-time cousins.
- `reflexion-self-critique` — single-agent iterative alternative.
- `autogen-multi-agent`,
  `crewai-hierarchical-agents` — frameworks that host debate
  workflows.
- `constitutional-ai` — training-time alignment cousin using an
  AI critic.

## Run

```
python techniques/multi-agent-debate/python/multi_agent_debate.py
Rscript techniques/multi-agent-debate/r/multi_agent_debate.R
```

**Refs:** Du, Y. et al. "Improving factuality and reasoning in language models through multiagent debate." *arXiv:2305.14325*, 2023; Liang, T. et al. "Encouraging divergent thinking in large language models through multi-agent debate." *EMNLP*, 2024.

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
