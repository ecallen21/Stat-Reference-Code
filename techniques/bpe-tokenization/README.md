# Byte-Pair Encoding — BPE Tokenization (Reference §47.159)

Gage (1994, data compression); Sennrich, Haddow & Birch (2016,
NMT). Learns a **merge table** from a corpus:

    1. Start with byte / character-level vocabulary.
    2. Count adjacent-symbol pairs; merge the most frequent one.
    3. Repeat until target vocab size or merge budget is reached.

Encoding: repeatedly apply learned merges to the input string.
Yields **sub-word units** that generalise to unseen / rare words.
Backbone of GPT (byte-level BPE), RoBERTa, T5 tokenizers.

## Files

- `python/bpe_tokenization.py` — from-scratch BPE learner + encoder
  on a 13-word morphology-rich corpus (`lower / lowest`, `wide /
  wider / widest`, `narrow / narrower / narrowest`):
  - 5 merges: `newest` → 7 tokens (still char-level).
  - 20 merges: `newest` → 2 tokens (`newe` + `st</w>`).
  - **50 merges**: `newest` → **1 token**; `widen` → 2 tokens;
    unseen `narrowly` → `narrow + l + y + </w>`.
  - Compression: **84.5 %** fewer tokens vs pure char-level.
- `r/bpe_tokenization.R` — `tokenizers.bpe::bpe` (YouTokenToMe /
  SentencePiece-style).

## When to use

- **Neural sequence models** (LLMs, MT, ASR) — sub-word tokens
  balance vocab size vs sequence length.
- **Low-resource languages** and morphologically rich languages —
  no OOV problem, robust to rare words.
- **Cross-lingual / multilingual** models — sub-word units
  transfer across related languages.

## When NOT to use

- **Character- or word-level tokenisation is enough** (small
  domain, tiny vocab).
- **Structured domains** with natural token boundaries (code
  ASTs, formulae).
- **When exact reversibility across whitespace / punctuation is
  needed** — byte-level BPE handles this; word-level BPE does not.

## Assumptions & caveats

- **Vocab size** — 8k-64k typical; larger reduces sequences but
  bloats embedding matrices.
- **Byte-level (GPT-2)** avoids UTF-8 issues; word-level (original
  Sennrich) is language-aware.
- **Deterministic merging** — the same corpus always yields the
  same merges given a fixed random tiebreak.
- **Sub-optimal per-string**: greedy merging is a heuristic;
  unigram-LM tokenisers (SentencePiece unigram) are more
  principled.

## Related in this repo

- `word2vec-skipgram`, `topic-modeling-lda` — text-representation
  cousins.
- `transformer-encoder`, `transformer-decoder`,
  `attention-mechanism` — the sequence models BPE feeds.
- `bertscore-chrf-metrics` — sub-word-based similarity metrics.

## Run

```
python techniques/bpe-tokenization/python/bpe_tokenization.py
Rscript techniques/bpe-tokenization/r/bpe_tokenization.R
```

**Refs:** Gage, P. "A new algorithm for data compression." *C Users Journal* 12(2), 1994; Sennrich, R., Haddow, B. & Birch, A. "Neural machine translation of rare words with subword units." *ACL*, 2016; Kudo, T. & Richardson, J. "SentencePiece: A simple and language independent subword tokenizer." *EMNLP*, 2018.

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
