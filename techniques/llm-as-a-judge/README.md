# LLM-as-a-Judge (Reference §47.216)

Zheng et al. (2023). Use a strong LM (GPT-4, Claude 3.5) as an
automated judge for open-ended response quality:

- **Single-answer grading**: score on 1-10.
- **Pairwise comparison**: which of A / B is better?
- **Reference-based**: compare to a gold answer.

Cheaper than human eval, correlates ~0.8 with crowd preferences.
**Known biases**: POSITION bias (favours A in A/B), VERBOSITY
bias, SELF-preference (judge favours its own family).

## Files

- `python/llm_as_a_judge.py` — toy judge scoring 3 responses to
  "Explain what an eigenvalue is." on 1-10 + pairwise + position-
  bias check:
  - Detailed response with example: **score 7 / 10**.
  - Brief correct response: 3.7 / 10.
  - Vague response: 3.7 / 10.
  - Pairwise 1-vs-2 = A wins; **position-swap agreement = 1.00**
    (no bias in this toy).
- `r/llm_as_a_judge.R` — no R port; recommends MT-Bench,
  alpaca-eval, chatbot-arena-leaderboard.

## When to use

- **Open-ended text quality eval** where automated metrics
  (BLEU, ROUGE) are inadequate.
- **RLHF reward model comparisons** — judge-based win-rate.
- **Rapid iteration** — cheap to run vs paid crowd workers.

## When NOT to use

- **Objective right-or-wrong tasks** — exact-match or task-
  specific metrics are cheaper and less biased.
- **Very small models as judges** — < 30 B params LMs correlate
  poorly with humans.
- **Regulated / legal contexts** where crowd or expert eval is
  required.

## Assumptions & caveats

- **Position bias** — always run pairwise in both orders.
- **Verbosity bias** — normalise for length or explicitly penalise.
- **Self-preference** — GPT-4 rates GPT-4 highly; use multiple
  judge families for reliability.
- **Prompt sensitivity** — small wording changes shift scores by
  points.
- **Not a substitute for domain experts** — use for coarse-grain
  filtering.

## Related in this repo

- `rlhf-preferences`,
  `dpo-direct-preference-optimization` — training-side
  applications.
- `process-reward-model-prm`,
  `best-of-n-sampling` — inference-time consumers of judge
  scores.
- `mmlu-benchmark-eval`, `ragas-rag-evaluation` — sibling LM
  evaluations.
- `multi-agent-debate` — multi-judge ensemble idea.

## Run

```
python techniques/llm-as-a-judge/python/llm_as_a_judge.py
Rscript techniques/llm-as-a-judge/r/llm_as_a_judge.R
```

**Refs:** Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS Datasets*, 2023.

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
