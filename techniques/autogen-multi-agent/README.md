# AutoGen — Multi-Agent Conversation Framework (Reference §47.207)

Wu et al. (2023, Microsoft). Framework for composing LLMs + tools
+ humans as **conversable agents**. Core primitives:

- **UserProxyAgent** — executes code / calls tools on behalf of
  the human.
- **AssistantAgent** — LM-powered planner / writer.
- **GroupChatManager** — coordinates a group chat among agents.

Agents exchange messages, each has its own system prompt, memory
and toolset. Enables patterns like "planner + coder + reviewer"
without hand-rolling message routing.

## Files

- `python/autogen_multi_agent.py` — toy 3-agent workflow
  (planner + coder + reviewer) on "write a Python function to
  add two numbers":
  - Round 1: planner emits step-by-step plan.
  - Round 2: coder emits Python function.
  - Round 3: reviewer signals **TASK COMPLETE**.
  - 4-message conversation, no message boilerplate.
- `r/autogen_multi_agent.R` — no R port; recommends `pyautogen`,
  `autogenhub` / `ag2`.

## When to use

- **Multi-step workflows** with distinct specialist roles.
- **Human-in-the-loop** — UserProxy can pause for approval.
- **Tool-use pipelines** — code exec, web browsing, file I/O
  built in.

## When NOT to use

- **Single-agent tasks** — plain function-calling is enough.
- **Latency-critical** production — agent turn-taking adds
  many LLM round-trips.
- **Simple queries** — orchestration overhead > single-call cost.

## Assumptions & caveats

- **Group-chat routing** — the manager picks the next speaker;
  policy determines quality.
- **Termination** — needs a clear stop signal ("TASK COMPLETE",
  max-turns, or user interrupt).
- **Cost / latency** compound with agent count.
- **Message context grows** — long conversations require
  summarisation.

## Related in this repo

- `crewai-hierarchical-agents` — task-DAG-first alternative.
- `react-reasoning-acting`,
  `function-calling-openai` — per-agent tool-use primitives.
- `multi-agent-debate` — a specific AutoGen-style workflow.
- `reflexion-self-critique` — single-agent memory alternative.

## Run

```
python techniques/autogen-multi-agent/python/autogen_multi_agent.py
Rscript techniques/autogen-multi-agent/r/autogen_multi_agent.R
```

**Refs:** Wu, Q. et al. "AutoGen: Enabling next-gen LLM applications via multi-agent conversation." *arXiv:2308.08155*, 2023.

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
