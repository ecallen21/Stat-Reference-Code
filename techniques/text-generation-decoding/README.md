# Text Generation Decoding (Reference §25.x extra)

Given an autoregressive language model `p(x_{t+1} | x_{1:t})`, choose the next
token. The **quality vs diversity vs speed** trade-off is entirely a decoding
decision — the trained model is fixed.

## Strategies

| Strategy | Formula | Behaviour |
|---|---|---|
| **Greedy** | `argmax p(x_{t+1} | …)` | deterministic, high likelihood, repetitive |
| **Beam search** | maintain top-B log-prob prefixes | deterministic, higher likelihood than greedy, still repetitive for long-form |
| **Temperature** | `softmax(logits / T)` | `T < 1` sharpens (more greedy), `T > 1` flattens |
| **Top-k** | keep k highest-p tokens, sample | limits vocab per step; k in ~40–100 typical |
| **Top-p / nucleus** | smallest set with cumulative mass ≥ p; sample | adapts vocab per step; p in ~0.9–0.95 typical |
| **Typical decoding** | (Meister 2022) prefer tokens with entropy near expected | reduces both very-low- and very-high-p outliers |
| **Min-p** | keep tokens with `p(x) ≥ p_min · max_p` | lightweight alternative to top-p |
| **MBR** (minimum Bayes risk) | sample many; return the one minimising expected task loss | quality-max at cost of many samples |
| **Speculative decoding** | small draft model + verify with big model | ~2-3× speed-up, same distribution as the big model |

## When to use

- **Deterministic / factoid QA**: greedy or low-temperature.
- **Translation / summarisation**: beam (B=4-8) with length normalisation.
- **Open-ended chat / creative writing**: top-p 0.9 with temperature 0.7–1.0.
- **Code generation**: greedy or low-temperature top-p.
- **Reasoning chains** (chain-of-thought, tree-of-thought): temperature 0.5–0.7 with self-consistency vote.
- **Latency-critical**: speculative decoding + KV-cache reuse.

## Files

- `python/text_generation_decoding.py` — from-scratch greedy, temperature sampling, top-k, top-p (nucleus), and beam search on a toy bigram LM (12-token vocab). Demo shows:
  - **Greedy** → deterministic loop `[0, 1, 2, 3, 4, 0, 1, 2, 3]`.
  - **Temperature** at 0.5 / 1.0 / 1.5 → progressively more diverse.
  - **Top-k (k=3)** and **top-p (p=0.9)** → constrained sampling.
  - **Beam (B=4)** → highest-log-prob completion.
- `r/text_generation_decoding.R` — `reticulate` + `transformers.pipeline('text-generation', do_sample=T, temperature=…, top_k=…, top_p=…, num_beams=…)`.

## Assumptions & caveats

- **Greedy / beam** collapse to repetition on long-form generation — length penalty and n-gram-blocking mitigate but don't fully fix.
- **Top-p and temperature interact** — tuning both together is standard; a common recipe is `T = 0.7, top_p = 0.9`.
- **Beam width > 8** rarely helps; beam-length normalisation matters more than width.
- **Sampling without seed = non-reproducible** — always set a seed for evaluation runs.
- **KV-cache** — reuse attention Ks and Vs across decoding steps; essential for real-time inference on transformer decoders.
- **Presence / frequency penalties** discourage repeated tokens; standard in chat APIs.
- **Log-probs** — request them from the API for calibration analysis / re-ranking.

## Related in this repo

- `transformer-decoder`, `ngram-language-model` — the LMs these strategies decode from.
- `bleu-rouge-eval`, `bertscore-chrf-metrics` — how to evaluate what you generated.
- `sentence-similarity` — semantic re-ranking after sampling.
- `contrastive-learning` — modern reward-model / RLHF training uses sampled generations.

## Run

```
python techniques/text-generation-decoding/python/text_generation_decoding.py
Rscript techniques/text-generation-decoding/r/text_generation_decoding.R
```

**Refs:** Fan, A., Lewis, M. & Dauphin, Y. "Hierarchical neural story generation." *ACL*, 2018 (top-k); Holtzman, A. et al. "The curious case of neural text degeneration." *ICLR*, 2020 (nucleus); Leviathan, Y., Kalman, M. & Matias, Y. "Fast inference from transformers via speculative decoding." *ICML*, 2023.

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
