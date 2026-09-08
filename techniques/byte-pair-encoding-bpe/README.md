# Byte-Pair Encoding (BPE) (Reference §47.83)

Sennrich, Haddow & Birch (2016), adapted from Gage (1994). Learn a
subword vocabulary by iteratively merging the most-frequent
adjacent symbol pair in a training corpus. Start with characters;
each merge adds one entry to the vocabulary. Handles morphology
and OOV gracefully — rare words fall back to characters. The
workhorse tokeniser of GPT / BERT / LLaMA families (GPT-2 uses
byte-level BPE over UTF-8 bytes for zero unknowns).

## Files

- `python/byte_pair_encoding_bpe.py` — from-scratch BPE
  training + apply. Demo (corpus `low`×5, `lowest`×2, `newer`×6,
  `wider`×3, `new`×4):
  - 4 merges: learns `er</w>`, `low`, `new`
  - 10 merges: full-word tokens `low`, `new`; OOV `lowering` →
    `low + er + i + n + g + </w>`
  - 20 merges: all training words become single tokens.
- `r/byte_pair_encoding_bpe.R` — `text2vec::bpe`,
  `tokenizers` (R); `tokenizers` (Hugging Face), `subword-nmt`,
  from-scratch (Python).

## When to use

- **LLM / seq2seq pre-training** — the default subword tokeniser.
- **Morphologically-rich languages** — Turkish, Finnish, Hungarian.
- **Rare / novel word handling** — OOVs decompose into learned
  subwords.
- **Compressed vocabulary** — 30–50 k tokens covers ~all real text.

## When NOT to use

- **Character-level tasks** requiring exact character alignment.
- **Very small corpora** — merges dominated by noise; use fixed
  vocabulary.
- **Domain-specific technical text** without co-adapted vocabulary —
  BPE trained on general text splits terms awkwardly.

## Assumptions & caveats

- **Merge ORDER matters** — apply merges in learned order at
  tokenisation time.
- **Whitespace handling** — pre-tokenise on whitespace or use
  byte-level BPE for full generality.
- **Vocabulary size** trade-off — larger vocab → fewer tokens per
  sequence, more parameters.
- **Deterministic** — no randomness in BPE learning.
- **Alternatives**: WordPiece (BERT), Unigram-LM (SentencePiece),
  BBPE (byte-level, GPT-2).

## Related in this repo

- `text-preprocessing`, `text-preprocessing-pipeline`,
  `tfidf-bm25`, `word-embeddings`, `sentence-similarity` — NLP
  preprocessing / representation.
- `transformer-encoder`, `transformer-decoder`,
  `masked-language-modeling`, `text-generation-decoding` — models
  that consume BPE tokens.
- `min-hash-lsh`, `record-linkage`, `string-similarity` — related
  string / hashing techniques.
- `feature-hashing`, `hyperloglog-cardinality` — hashing cousins.

## Run

```
python techniques/byte-pair-encoding-bpe/python/byte_pair_encoding_bpe.py
Rscript techniques/byte-pair-encoding-bpe/r/byte_pair_encoding_bpe.R
```

**Refs:** Sennrich, R., Haddow, B. & Birch, A. "Neural machine translation of rare words with subword units." *ACL*, 2016; Gage, P. "A new algorithm for data compression." *C/C++ Users Journal*, 1994; Kudo, T. & Richardson, J. "SentencePiece: A simple and language independent subword tokenizer." *EMNLP demo*, 2018.

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
