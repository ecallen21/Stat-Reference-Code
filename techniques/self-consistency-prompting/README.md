# Self-Consistency Prompting (Reference §47.91)

Wang et al (2022). Instead of taking the FIRST chain-of-thought
answer, sample K reasoning paths at temperature T > 0 and return
the MAJORITY VOTE:

    y* = mode({ y_k : (r_k, y_k) ~ p(reason, answer | prompt), k = 1..K }).

Independent errors cancel; correct paths agree. Consistently
beats greedy CoT on math and multi-hop reasoning benchmarks.

## Files

- `python/self_consistency_prompting.py` — simulates stochastic
  reasoning agents whose 4-step chains are all-correct with prob
  `0.7^4 ≈ 0.24`; majority-vote lifts accuracy dramatically. Demo
  (500 problems):
  - K = 1  → 0.234 (single chain)
  - K = 5  → 0.338
  - K = 20 → 0.656
  - K = 40 → **0.892** (nearly 4× the single-chain rate).
- `r/self_consistency_prompting.R` — LLM tooling lives in Python;
  `litellm`, `lmql`, `vllm` batching.

## When to use

- **LLM math / logic / multi-hop QA** — any task where diverse
  reasoning paths converge on the same answer.
- **Code-generation with unit-test verification** — sample K
  candidates, keep those that pass tests.
- **Answer-consistency confidence** — spread across K samples =
  calibrated uncertainty signal.

## When NOT to use

- **Deterministic tasks** — single greedy CoT suffices.
- **Open-ended generation** (essays, poems) — no unique correct
  answer to vote on.
- **Latency-critical serving** — K× the compute.
- **Correlated errors** — if all chains share the same
  misunderstanding, voting doesn't help.

## Assumptions & caveats

- **Temperature > 0** — else all samples identical.
- **K trade-off** — accuracy saturates around K = 40–100 for math
  benchmarks (paper: GSM8K).
- **Answer extraction** — need canonicalisation to compare
  answers exactly (strip units, normalise fractions).
- **Cost**: K× inference; use adaptive stopping (majority-vote
  crossings) to prune.

## Related in this repo

- `chain-of-thought-reasoning`, `tree-of-thoughts`,
  `in-context-learning`, `retrieval-augmented-generation` — LLM
  reasoning cousins.
- `deep-ensembles`, `mc-dropout`, `bayesian-neural-network`,
  `swag` — uncertainty via sampling.
- `bagging-oob`, `nonparametric-bootstrap`,
  `stability-selection` — resampling analogues.
- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  post-training LLM alignment.

## Run

```
python techniques/self-consistency-prompting/python/self_consistency_prompting.py
Rscript techniques/self-consistency-prompting/r/self_consistency_prompting.R
```

**Refs:** Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A. & Zhou, D. "Self-consistency improves chain of thought reasoning in language models." *ICLR*, 2023.

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
