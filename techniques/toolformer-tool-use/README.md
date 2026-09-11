# Toolformer — Self-Taught Tool Use (Reference §47.203)

Schick et al. (2023, NeurIPS). Fine-tunes an LM to insert
API-call tokens INTO its own generation:

    "The population of Kansas is [Calculator(2 + 3) = 5] approximately..."

Training pipeline:

1. Prompt the LM to sample candidate API insertion points.
2. Execute the API; keep the insertion if it **reduces
   perplexity** on the subsequent context.
3. Fine-tune the LM on the filtered augmented sequences.

Requires no human demonstration of tool use — self-supervised.

## Files

- `python/toolformer_tool_use.py` — toy Toolformer filter pass
  on 3 candidate API insertions (Calculator, Wiki):
  - Calculator(2 + 3)→5 reduces perplexity by 1.5 → **KEEP**
  - Calculator(10 * 10) and Wiki(python) irrelevant → **DROP**
  - Final augmented text contains only the useful call.
- `r/toolformer_tool_use.R` — no R port; recommends
  `conceptofmind/toolformer`, `lucidrains/toolformer-pytorch`.

## When to use

- **Fine-tuning a base LM to use tools** without human
  demonstrations.
- **Domain-specific APIs** where instruction-tuned function
  calling misroutes.
- **When calibrated tool use** at token level matters (e.g.
  "insert a citation here").

## When NOT to use

- **When function-calling APIs are enough** — hosted models
  (GPT-4, Claude) come with structured tool use out of the box.
- **Very small models** — can't reliably decide when to call.
- **Tools with side effects** — perplexity filtering doesn't
  reason about consequences.

## Assumptions & caveats

- **Perplexity threshold** — cutoff for keep/drop decisions;
  tuned per-task.
- **API budget** — every candidate needs a real API call at
  filtering time.
- **Long-form perplexity** window — need enough post-context to
  measure impact.
- **Not causal** — Toolformer learns correlations between tool
  results and next-token probs, not consequences.

## Related in this repo

- `react-reasoning-acting`,
  `function-calling-openai` — sibling tool-use paradigms.
- `retrieval-augmented-generation` — a specialised tool
  (retriever) baked in.
- `instruction-tuning-flan`, `constitutional-ai` — training-
  time cousins that add different capabilities.

## Run

```
python techniques/toolformer-tool-use/python/toolformer_tool_use.py
Rscript techniques/toolformer-tool-use/r/toolformer_tool_use.R
```

**Refs:** Schick, T. et al. "Toolformer: Language models can teach themselves to use tools." *NeurIPS*, 2023.

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
