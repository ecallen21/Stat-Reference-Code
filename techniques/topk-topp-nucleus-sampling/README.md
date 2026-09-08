# Top-k / Top-p (Nucleus) Sampling (Reference §47.110)

Fan, Lewis & Dauphin (2018, top-k); Holtzman et al (2020, top-p /
nucleus). Truncate the next-token distribution before sampling:

- **Top-k**: restrict to the K most-probable tokens.
- **Top-p (nucleus)**: restrict to the SMALLEST set whose cumulative
  probability ≥ p (adaptive vocab size).

Temperature `T` rescales logits before softmax: `pᵢ ∝ exp(logitᵢ/T)`.

## Files

- `python/topk_topp_nucleus_sampling.py` — from-scratch softmax /
  top-k / top-p samplers + diversity metrics on a long-tail Zipf-
  like distribution. Demo (V=200):
  - top-k K=1  → 1 unique token in 5 000 draws (greedy collapse)
  - top-k K=20 → 20 unique
  - top-p p=0.99 → 45 unique (adaptive vocab)
  - Temperature 0.5 collapses entropy to 0.33; T=1.5 raises to 3.15.
- `r/topk_topp_nucleus_sampling.R` — no first-class R decoding;
  `transformers.generation` in Python.

## When to use

- **Open-ended text generation** — creative writing, dialogue,
  translation.
- **Any autoregressive sampler** — audio, code, molecular strings.
- **Avoid repetition** without full temperature — nucleus is
  content-adaptive.

## When NOT to use

- **Constrained decoding** where every token must be valid — use
  logit masking + greedy or beam search.
- **Extractive tasks** (span QA, translation with strict
  targets) — greedy / beam typically better.
- **When you want the argmax answer** — greedy or beam.

## Assumptions & caveats

- **Temperature < 1** peakier; > 1 flatter; typically 0.7-1.0.
- **top-k vs top-p** — top-p adapts to distribution shape; top-k
  is simpler + faster.
- **Repetition penalties** (Keskar 2019) reduce loops beyond
  sampling tricks.
- **Beam-diverse variants**: diverse beam search (Vijayakumar 2018)
  and typical-p sampling (Meister 2023) address related issues.

## Related in this repo

- `text-generation-decoding`, `speculative-decoding`,
  `chain-of-thought-reasoning`, `self-consistency-prompting`,
  `tree-of-thoughts`, `retrieval-augmented-generation` — decoding
  / prompting cousins.
- `transformer-decoder`, `transformer-encoder`,
  `masked-language-modeling`, `text-generation-decoding` — model
  families the decoding runs on.
- `rlhf-preferences`, `dpo-direct-preference-optimization` —
  alignment methods that shape the distribution being sampled.

## Run

```
python techniques/topk-topp-nucleus-sampling/python/topk_topp_nucleus_sampling.py
Rscript techniques/topk-topp-nucleus-sampling/r/topk_topp_nucleus_sampling.R
```

**Refs:** Fan, A., Lewis, M. & Dauphin, Y. "Hierarchical neural story generation." *ACL*, 2018; Holtzman, A., Buys, J., Du, L., Forbes, M. & Choi, Y. "The curious case of neural text degeneration." *ICLR*, 2020.

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
