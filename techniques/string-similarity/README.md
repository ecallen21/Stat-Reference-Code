# String Similarity (Reference §25.9)

Quantify how similar two strings are — for record linkage, typo detection,
fuzzy join, deduplication, or approximate search.

## Metrics

| Metric | Formula sketch | Best at |
|---|---|---|
| **Levenshtein** | min inserts + deletes + substitutions | general edit distance |
| **Damerau-Levenshtein** | + adjacent transposition as one op | keyboard typos |
| **Jaro** | matching-window + transpositions, normalised | short name matching |
| **Jaro-Winkler** | Jaro + bonus for common prefix | person / company names |
| **LCS ratio** | longest common subsequence length / max(|a|, |b|) | free-text alignment |
| **Jaccard on char n-grams** | |A ∩ B| / |A ∪ B| | long strings, tolerant of insertions |
| **Cosine on char n-grams** | dot / norms of n-gram counts | as above, respects repetition |
| **Soundex / Metaphone / NYSIIS** | phonetic hashing | name matching across spelling variants |

## When to use

- **Record linkage** — merging customer records across systems with typos.
- **Fuzzy join** — matching product names, addresses across databases (`fuzzyjoin` in R, `dedupe` in Python).
- **Spell-correction** — closest-vocab-word retrieval (BK-tree of Levenshtein).
- **Duplicate / near-duplicate detection** — MinHash + Jaccard for very large corpora.
- **Approximate search** — Levenshtein automata for indexed fuzzy string search.

## Files

- `python/string_similarity.py` — from-scratch Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, Jaccard (char-tri-gram), cosine (char-tri-gram). Demo on a small pair table: `form` ↔ `from` = Lev 2 / Damerau 1 (transposition), Martha ↔ Marhta same, `Elisabeth` ↔ `Elizabeth` = Lev 1 with JW 0.971 (long shared prefix), apple ↔ orange completely dissimilar.
- `r/string_similarity.R` — `stringdist::stringdist`, `stringdist::stringsim`, `fuzzyjoin::stringdist_inner_join`, `RecordLinkage`, `phonics::soundex / metaphone`.

## Assumptions & caveats

- **Edit distance is O(n·m)** — for millions of pairs, use blocking (index by phonetic hash / n-gram MinHash) then compare only within blocks.
- **Jaro-Winkler is asymmetric to prefix errors** — good for names ("Robert" vs "Robret") but not for suffix errors.
- **Character n-grams miss word order** — "New York City" vs "City of New York" have identical bag-of-trigrams.
- **Normalise before comparing** — case, punctuation, diacritics (unicode NFKD). Otherwise "café" vs "cafe" scores mid-range.
- **Domain-specific pre-processing** helps a lot — strip company suffixes ("Corp", "Inc"), expand abbreviations, standardise addresses (`usaddress`, `libpostal`).
- **Semantic vs surface** — string similarity is entirely surface. For semantic similarity ("automobile" vs "car"), use dense embeddings (`word-embeddings`, sentence-transformers).

## Related in this repo

- `text-preprocessing` — normalisation before comparison.
- `word-embeddings` — semantic alternative.

## Run

```
python techniques/string-similarity/python/string_similarity.py
Rscript techniques/string-similarity/r/string_similarity.R
```

**Refs:** Levenshtein, V.I. "Binary codes capable of correcting deletions, insertions, and reversals." *Soviet Physics Doklady* 10(8), 707–710, 1966; Winkler, W.E. "String comparator metrics and enhanced decision rules in the Fellegi-Sunter model of record linkage." *ASA Section on Survey Research Methods*, 1990; Manning et al., *Introduction to Information Retrieval*, Cambridge UP, 2008.

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
