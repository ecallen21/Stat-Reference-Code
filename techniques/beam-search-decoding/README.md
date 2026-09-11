# Beam-Search Decoding (Reference §47.158)

Reddy (1977, Hearsay-II); Lowerre (1976, Harpy). Approximate
best-first search over a discrete sequence-generation model. Keep
top B partial hypotheses (**beams**) at each step:

    for t = 1..T:
        for each active beam h:
            expand by all V symbols; score h||v with log p(v | h)
        keep top B highest-scoring beams (with length norm)

O(T · V · B) instead of exact O(V^T). Length normalisation avoids
favouring shorter sequences. Backbone of neural MT / summarisation
inference.

## Files

- `python/beam_search_decoding.py` — pure-Python beam search on a
  toy 8-token bigram LM designed so that greedy gets stuck at a
  suboptimum:
  - **Greedy / B = 1**: seq = [0, 1, 1, 1, ..., 1] (length 16),
    log_prob = **−23.13**
  - **B ≥ 2**: seq = [0, 4, 5, 6, 7] (length 5),
    log_prob = **−2.18** — >10× better in log-space.
- `r/beam_search_decoding.R` — pure-R implementation on the same
  transition matrix.

## When to use

- **Sequence generation** (NMT, ASR, summarisation, image
  captioning, code generation).
- **Anywhere autoregressive decoding is expensive but exact search
  is infeasible**.
- **Structured prediction** where a scoring model factorises over
  steps.

## When NOT to use

- **When exact search is tractable** (very short, small-vocab
  sequences) — Viterbi / DP is optimal.
- **Diverse output generation** — beam search collapses on a
  single mode; use nucleus / top-k / diverse beam search.
- **When calibrated log-probs matter** — beam search over-favours
  short sequences without length-normalisation.

## Assumptions & caveats

- **Length penalty α** (Wu 2016 GNMT): score / (length + 5)^α /
  (5 + 1)^α; typical α = 0.6-1.0.
- **Length-normalisation trap** — too much penalty inflates
  low-quality long sequences.
- **Beam width B** — 4-10 typical for MT; larger rarely helps
  and can hurt (label-bias).
- **End-of-sequence token** essential; without it beams grow
  indefinitely.

## Related in this repo

- `viterbi-algorithm` — exact best-path DP (Viterbi = beam
  search with B = ∞).
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder` — the sequence models beam search
  usually decodes.
- `abstractive-summarization`, `question-answering`,
  `relation-extraction` — downstream applications.

## Run

```
python techniques/beam-search-decoding/python/beam_search_decoding.py
Rscript techniques/beam-search-decoding/r/beam_search_decoding.R
```

**Refs:** Reddy, D. R. "Speech understanding systems: a summary of results of the five-year research effort." *CMU-CS-77-CS-215*, 1977; Lowerre, B. T. "The Harpy Speech Recognition System." *PhD thesis, CMU*, 1976; Wu, Y. et al. "Google's Neural Machine Translation System." *arXiv:1609.08144*, 2016; Vijayakumar, A. K. et al. "Diverse beam search." *arXiv:1610.02424*, 2016.

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
