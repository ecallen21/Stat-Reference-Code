# Benchmark Contamination Detection (Reference §47.219)

Sainz et al. (2023); Golchin & Surdeanu (2023, Time Travel in
LLMs); Xu et al. (2024). Common tests:

1. **Guided-completion test**: give the LM the first half of a
   benchmark example and see if it exactly completes it.
2. **Membership inference**: compare loss / perplexity on
   training-time vs held-out examples.
3. **Rephrase test**: paraphrase the example — a memorising
   model drops accuracy, a reasoning model doesn't.
4. **N-gram overlap**: search the training data for the exact
   benchmark strings.

## Files

- `python/contamination_detection.py` — three-way toy detector on
  a 'memorised' vs 'novel' benchmark:
  - **Guided-completion**: 2/2 memorised → ✓ MATCH; 2/2 novel →
    ✗ miss (correctly identifies memorisation).
  - **Membership inference**: train ppl = 1.5 vs held-out 5.0
    (gap 3.5 flags contamination).
  - **Rephrase test**: orig acc 1.00 → paraphrased 0.00 (drop
    1.0 confirms memorisation).
- `r/contamination_detection.R` — no R port; recommends
  `lm-eval-harness` contamination flags, `llm-contamination`.

## When to use

- **Before trusting a benchmark score** — check whether the LM
  saw the eval set.
- **Comparing LMs from different training-data eras** —
  contamination invalidates fair comparison.
- **Regulatory audits** — proof of eval integrity.

## When NOT to use

- **When training data is open** — direct n-gram search is
  cheaper.
- **Very small LMs** — memorisation is limited; detection is
  noisy.
- **Extremely long benchmarks** — guided completion becomes
  ambiguous.

## Assumptions & caveats

- **Prompt / template sensitivity** — small variations mask
  memorisation.
- **Rephrase quality** matters — LM paraphrases can leak the
  answer.
- **False positives** — highly-predictable benchmark text can
  look 'memorised' even when it isn't.
- **Uncontaminated held-out** essential for membership
  inference.

## Related in this repo

- `mmlu-benchmark-eval`, `ragas-rag-evaluation`,
  `llm-as-a-judge` — eval methods that need contamination
  vetting.
- `data-drift-detection` — related but different (input
  distribution shift).
- `model-cards`, `datasheets-for-datasets` — documentation
  cousins.

## Run

```
python techniques/contamination-detection/python/contamination_detection.py
Rscript techniques/contamination-detection/r/contamination_detection.R
```

**Refs:** Sainz, O. et al. "NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark." *Findings EMNLP*, 2023; Golchin, S. & Surdeanu, M. "Time travel in LLMs: Tracing data contamination in large language models." *ICLR*, 2024; Xu, C. et al. "Benchmarking benchmark leakage in large language models." *arXiv:2404.18824*, 2024.

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
