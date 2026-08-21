# Language Detection (Reference §25.10)

Identify the language of a text. Classical Cavnar-Trenkle (1994) approach:

1. **Build a per-language profile**: rank the top-K most frequent character
   n-grams (n = 1..5) in a training corpus for each language.
2. **For a new text**: compute its own top-K n-gram profile.
3. **Score each language** by the **out-of-place distance** — sum of rank
   differences (with a K penalty for missing n-grams).
4. Pick the smallest.

## Alternatives

- **Naive Bayes on character n-grams** — Google CLD2's underlying model.
- **Neural CLD3** — small feed-forward network; fast and accurate.
- **FastText `lid.176.bin`** — 176 languages, ~2 MB, near-SOTA accuracy.
- **Script + character-run heuristics** — cheap first-pass (Latin vs Cyrillic vs CJK) before finer discrimination.

## When to use

- **Routing / preprocessing** — apply the right tokeniser / stemmer / model per language.
- **Cleaning multilingual dumps** — filter Common Crawl by desired languages.
- **Detecting language mixes** — sentence-level detection reveals code-switching (`langdetect_profile.py`).

## Files

- `python/language_detection.py` — from-scratch Cavnar-Trenkle profiles + out-of-place scoring. Toy demo with English / Spanish / German training paragraphs classifies 5/6 held-out test snippets correctly; the miss is a very short (4-word) Spanish phrase misclassified as German — the training corpus is tiny.
- `r/language_detection.R` — `cld2::detect_language`, `cld3::detect_language`, `textcat::textcat`, `fastTextR`.

## Assumptions & caveats

- **Short inputs are hard** — < 20 characters can be ambiguous even for FastText; return top-3 with confidences.
- **Related languages** (Spanish / Portuguese / Italian; Norwegian / Danish; Serbian / Croatian) are difficult; character n-grams are the right feature but need much more training text.
- **Code-switching** — sentence-level detection is needed; document-level returns the dominant language.
- **URLs, code, emoji** — clean before detection; a JavaScript snippet with English identifiers is not English.
- **Bytecode / encoding** — always decode to Unicode first; language ID on mis-encoded bytes is meaningless.
- **Latency vs accuracy** — CLD2 is microseconds per doc; FastText is milliseconds but more accurate.

## Related in this repo

- `text-preprocessing` — the tokenisation step downstream of language ID.
- `text-classification`, `naive-bayes` — the general classifier machinery underlying most language ID.

## Run

```
python techniques/language-detection/python/language_detection.py
Rscript techniques/language-detection/r/language_detection.R
```

**Refs:** Cavnar, W.B. & Trenkle, J.M. "N-gram-based text categorization." *SDAIR*, 1994; Lui, M. & Baldwin, T. "langid.py: An off-the-shelf language identification tool." *ACL*, 2012; Joulin, A. et al. "Bag of tricks for efficient text classification." *EACL*, 2017 (FastText basis for lid.176).

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
