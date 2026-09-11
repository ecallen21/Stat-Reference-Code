# MMLU-Style Benchmark Evaluation (Reference §47.217)

Hendrycks et al. (2021, ICLR). Multiple-choice QA benchmark
spanning 57 academic + professional subjects. Standard evaluation
protocol:

- **5-shot** prompt (5 (Q, A) demonstrations from the same
  subject).
- Score by **next-token log-prob** over {A, B, C, D}; take
  argmax.
- Report **macro accuracy** (mean over subjects).

Widely used but has known issues: contamination (MMLU appears in
web-scraped corpora), letter bias (models over-select A/B), and
prompt sensitivity.

## Files

- `python/mmlu_benchmark_eval.py` — toy 6-question algebra
  subject with per-letter log-prob scoring:
  - **Ideal LM** (parses and solves): accuracy ≈ 0.5 on this
    ill-posed rule-based stub (chance = 0.25).
  - **Letter-A biased LM**: 0.0 (subject has no A-correct answers).
  - Shows the scoring protocol and letter-bias artifact.
- `r/mmlu_benchmark_eval.R` — no R port; recommends
  `lm-eval-harness`, `opencompass`, HELM.

## When to use

- **Standard LM benchmarking** across broad academic subjects.
- **Reporting** — MMLU is a commonly-cited headline metric.
- **Debugging** — subject-level scores localise LM strengths /
  weaknesses.

## When NOT to use

- **When your task doesn't match MMLU's format** (multiple-choice
  academic).
- **When contamination is a concern** — many modern LMs have seen
  MMLU. Check with contamination-detection.
- **For fine-grained reasoning** — MMLU rewards recall more than
  reasoning; use GPQA / MATH / HumanEval instead.

## Assumptions & caveats

- **Prompt sensitivity** — 5-shot template choice moves scores by
  1-3 pt.
- **Letter bias** — some LMs favour A/B; report per-letter
  accuracy.
- **Contamination** — MMLU is on the web; models trained after
  2020 likely saw it.
- **Macro vs micro averaging** — MMLU standard is macro (mean
  over subjects).

## Related in this repo

- `llm-as-a-judge` — orthogonal open-ended eval method.
- `contamination-detection` — check whether MMLU leaked.
- `ragas-rag-evaluation` — RAG-side eval.
- `chain-of-thought-reasoning`,
  `tree-of-thoughts-reasoning` — inference-time methods that
  can improve MMLU scores.

## Run

```
python techniques/mmlu-benchmark-eval/python/mmlu_benchmark_eval.py
Rscript techniques/mmlu-benchmark-eval/r/mmlu_benchmark_eval.R
```

**Refs:** Hendrycks, D. et al. "Measuring massive multitask language understanding (MMLU)." *ICLR*, 2021; Wang, Y. et al. "MMLU-Pro: A more robust and challenging multi-task language understanding benchmark." *arXiv:2406.01574*, 2024.

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
