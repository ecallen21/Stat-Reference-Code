# MinHash + LSH (Reference §47.76)

Broder (1997); Indyk & Motwani (1998). For sets A, B:

    P( min h(A) = min h(B) )  =  |A ∩ B| / |A ∪ B|  =  J(A, B).

K independent hashes give an unbiased Jaccard estimator with
variance ~1/K. **Locality-Sensitive Hashing** splits the K-length
signature into b bands of r rows; declare candidate pairs by any
band match — O(n) near-neighbour retrieval with tunable
recall/precision trade-off.

## Files

- `python/min_hash_lsh.py` — SHA-1 MinHash signatures + banded
  LSH index from scratch. Demo:
  - 3 short documents (Jaccard 0.78, 0, 0): K=256 gives Jaccard
    estimates 0.73, 0.00, 0.00 (|err| ≤ 0.05).
  - LSH on 200 random shingle sets with 5 planted near-duplicates
    (K=128, b=32, r=4): all 5 duplicate pairs recovered, only 14
    candidate pairs to check out of C(200, 2) = 19 900.
- `r/min_hash_lsh.R` — `LSHR`, `textreuse::minhash_generator`
  (R); `datasketch.MinHash`, `datasketch.MinHashLSH`, from-scratch
  (Python).

## When to use

- **Near-duplicate document / URL detection** at web scale.
- **Plagiarism / clone detection** (code, essays).
- **Nearest-neighbour retrieval** on Jaccard/Hamming metrics —
  ANN with tunable recall.
- **Set-similarity joins** over billions of sets (Spark, Beam).

## When NOT to use

- **Similarity metric ≠ Jaccard** — use SimHash (angular) or
  cosine LSH for embedding vectors.
- **Very small sets** — direct set intersection is cheap.
- **Exact recall required** — LSH is approximate; combine with
  post-filter that computes true Jaccard on candidates.

## Assumptions & caveats

- **K, b, r trade-off** — pick b, r to hit the desired
  probability curve `1 − (1 − s^r)^b` at the similarity threshold.
- **Hash quality** — cheap hashes (MurmurHash3, xxHash) are
  standard.
- **Shingling** — token / word / n-gram; choice affects Jaccard.
- **Weighted sets** — use weighted MinHash (ICWS).

## Related in this repo

- `hyperloglog-cardinality`, `bloom-filter-membership` —
  probabilistic data-structure family.
- `record-linkage`, `data-drift-detection` — similarity-based
  deduplication / drift.
- `feature-hashing` — Weinberger et al 2009 hashing trick.
- `tfidf-bm25`, `sentence-similarity`, `string-similarity` —
  text-similarity cousins.

## Run

```
python techniques/min-hash-lsh/python/min_hash_lsh.py
Rscript techniques/min-hash-lsh/r/min_hash_lsh.R
```

**Refs:** Broder, A.Z. "On the resemblance and containment of documents." *Compression and Complexity of Sequences*, 1997; Indyk, P. & Motwani, R. "Approximate nearest neighbors: towards removing the curse of dimensionality." *STOC*, 1998.

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
