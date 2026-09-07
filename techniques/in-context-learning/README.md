# In-Context Learning (ICL) (Reference §47.20)

Brown et al. (2020, GPT-3). Rather than fine-tuning, large LMs can
"learn" a task from a few input-output examples supplied in the
prompt:

    Q: 1 + 2 = ?
    A: 3
    Q: 5 + 4 = ?
    A: 9
    Q: 7 + 6 = ?
    A: <predict>

## Regimes

| Regime | # demos in prompt |
|---|---|
| Zero-shot | 0 (instructions only) |
| One-shot | 1 |
| Few-shot (K) | K, typically 4–32 |

Empirically, task accuracy scales **log-linearly** with K up to a
plateau.

## Theoretical view

Xie et al. (2022), Akyürek et al. (2023): under certain prompt
distributions, ICL is **equivalent to implicit Bayesian inference**
or **gradient descent** performed inside the transformer's forward
pass.

## Files

- `python/in_context_learning.py` — toy "transformer as OLS" proxy:
  fits OLS on K demos to predict for a new query, then evaluates
  over 500 random regression tasks. Demo (task: y = a·x + b, noise
  0.2): MSE drops from 8.1 (K=0) → 15.9 (K=1, high variance) → 9.75
  (K=2) → 0.03 (K=4) → 0.002 (K=32) — classic ICL curve.
- `r/in_context_learning.R` — `ellmer`, `chattr`, `gptstudio` (R);
  openai / anthropic / transformers / lm-eval-harness (Python).

## When to use

- **New tasks without training data** — a handful of demos + a good
  LM often beats a fine-tuned small model.
- **Rapid prototyping** — no infrastructure for training.
- **Task adaptation on the fly** — different demos per query.

## When NOT to use

- **Latency-sensitive** — long prompts (many demos) are expensive
  per-request; distill or fine-tune.
- **Very-hard reasoning** — even large LMs need chain-of-thought,
  tools, or fine-tuning.
- **Precision-critical numerics** — arithmetic and long-chain
  reasoning benefit from tools + verification, not raw ICL.

## Assumptions & caveats

- **Demo curation matters** — order, diversity, and format of
  examples change accuracy dramatically.
- **Distribution shift** — ICL fails when test items lie outside the
  demo distribution.
- **Position bias** — the last demos in the prompt dominate; shuffle
  demos.
- **Contamination** — ICL benchmarks are often in the training
  corpus; use held-out probes.
- **Model size** — the effect is strongest in large LMs; small
  models often do not exhibit meaningful ICL.

## Related in this repo

- `retrieval-augmented-generation` — orthogonal grounding
  technique.
- `text-generation-decoding`, `transformer-decoder`, `attention-mechanism`
  — LM machinery.
- `meta-learning-maml` — an explicit few-shot learner outside LMs.
- `knowledge-distillation` — compress a few-shot policy into a
  small model.

## Run

```
python techniques/in-context-learning/python/in_context_learning.py
Rscript techniques/in-context-learning/r/in_context_learning.R
```

**Refs:** Brown, T. et al. "Language models are few-shot learners." *NeurIPS*, 33: 1877-1901, 2020; Xie, S.M. et al. "An explanation of in-context learning as implicit Bayesian inference." *ICLR*, 2022; Akyürek, E. et al. "What learning algorithm is in-context learning?" *ICLR*, 2023.

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
