# CrewAI — Hierarchical Multi-Agent (Reference §47.208)

Moura (2024, CrewAI OSS). Structures multi-agent workflows around
a **crew** of specialised agents, a set of **tasks** with declared
inputs/outputs, and either **sequential** or **hierarchical**
process (where a manager agent routes tasks).

    Agent(role, goal, backstory, tools)
    Task(description, agent, expected_output, context=[prev_tasks])
    Crew(agents, tasks, process="sequential" | "hierarchical")

Contrasts with AutoGen's turn-taking chat: CrewAI's tasks have
explicit DAG dependencies and typed outputs.

## Files

- `python/crewai_hierarchical_agents.py` — toy 3-agent crew
  (researcher + writer + editor) writing a paragraph about
  Python's history:
  - **Sequential mode**: researcher → writer → editor.
  - **Hierarchical mode**: manager reads each task and routes to
    the right specialist by role keyword.
  - Both produce the same well-edited final article.
- `r/crewai_hierarchical_agents.R` — no R port; recommends
  `crewai`, `crewai-tools`.

## When to use

- **Well-defined pipelines** with named stages and typed outputs.
- **When manager-style routing** matters more than free-form
  chat.
- **Enterprise workflows** — audit trail per task, role-based
  agents.

## When NOT to use

- **Free-form open-ended tasks** — sequential DAG is too rigid.
- **Rapid prototyping** — AutoGen / langgraph faster to iterate.
- **Very fine-grained tool loops** — task granularity is
  workflow-level.

## Assumptions & caveats

- **Task DAG** must be authored explicitly (dependency = context
  list).
- **Manager routing** requires a competent LM to parse task
  descriptions.
- **State passing** through `context=` — verbose but explicit.
- **Tool integration** — CrewAI tools are role-scoped; less
  flexible than AutoGen's shared toolbox.

## Related in this repo

- `autogen-multi-agent` — turn-taking chat alternative.
- `multi-agent-debate` — a specific multi-agent pattern.
- `react-reasoning-acting`,
  `function-calling-openai` — per-agent tool primitives.
- `reflexion-self-critique`,
  `constitutional-ai` — single-agent iterative improvement.

## Run

```
python techniques/crewai-hierarchical-agents/python/crewai_hierarchical_agents.py
Rscript techniques/crewai-hierarchical-agents/r/crewai_hierarchical_agents.R
```

**Refs:** Moura, J. "CrewAI: Framework for orchestrating role-playing autonomous AI agents." *github.com/crewaiinc/crewai*, 2024.

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
