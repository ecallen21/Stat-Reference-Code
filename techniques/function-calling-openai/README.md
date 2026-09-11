# Function Calling / Tool Use (Reference §47.204)

OpenAI (2023, function calling); Anthropic (2024, tool use).
Instead of embedding tool calls in natural language (Toolformer /
ReAct), the LM emits **structured JSON calls** when it decides a
tool is needed:

    user: "What's the weather in Tokyo?"
    assistant: { "name": "get_weather",
                 "arguments": { "city": "Tokyo", "unit": "celsius" } }
    tool_result: { "temp": 18, "conditions": "cloudy" }
    assistant: "It's 18 °C and cloudy in Tokyo."

The tool schemas are declared in advance (JSON Schema); the LM
chooses among them by name and fills required arguments.

## Files

- `python/function_calling_openai.py` — toy declarative tool set
  (get_weather, calc) + rule-based LM. Four prompts:
  - "What's the weather in Tokyo?" → `get_weather({...})` →
    18 °C, cloudy.
  - "Calculate 42 · 17 + 5." → `calc({expression: "42*17+5"})` →
    719.
  - "Tell me a joke." → direct answer (no tool).
  - "How's the weather in Paris?" → `get_weather` → 12 °C, rain.
- `r/function_calling_openai.R` — no R port; recommends the
  `openai` / `anthropic` Python SDKs' native tool APIs.

## When to use

- **Any production LLM app with tools** — the standard 2024+
  pattern.
- **Structured downstream execution** — the JSON schema
  guarantees parseable output.
- **Multi-tool routing** — LM picks the right tool by name.

## When NOT to use

- **Free-form / open-ended tool use** — Toolformer's inline
  calls are more flexible.
- **Very small models** — need strong instruction following to
  emit valid JSON.
- **When you need chain-of-thought inside the tool call** —
  structured JSON hides the reasoning.

## Assumptions & caveats

- **Schema-first design** — every tool needs a JSON Schema
  declaration.
- **Validation is essential** — LMs occasionally emit malformed
  JSON.
- **Parallel tool calling** (OpenAI, Anthropic) — multiple tools
  in one turn.
- **Fine-tuned vs zero-shot** — Fine-tuned models (GPT-4-tools,
  Claude 3.5) far outperform zero-shot LMs on tool routing.

## Related in this repo

- `react-reasoning-acting`, `toolformer-tool-use` — alternative
  tool-use paradigms.
- `autogen-multi-agent`,
  `crewai-hierarchical-agents` — frameworks built on top of
  function calling.
- `retrieval-augmented-generation`, `self-rag` — RAG as a
  specialised tool.

## Run

```
python techniques/function-calling-openai/python/function_calling_openai.py
Rscript techniques/function-calling-openai/r/function_calling_openai.R
```

**Refs:** OpenAI. "Function calling and other API updates." *openai.com blog*, 2023; Anthropic. "Tool use with Claude." *anthropic.com docs*, 2024.

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
