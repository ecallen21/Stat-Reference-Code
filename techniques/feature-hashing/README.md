# Feature Hashing (Reference §41.10)

Weinberger et al. (2009). Map arbitrary string / categorical features
to a fixed `d`-dimensional vector via `hash(feature) mod d`. A
**signed** variant uses a second hash bit as `±1` so collisions are
unbiased in expectation.

## Why hash

- **Constant memory** regardless of vocabulary size.
- **No dictionary** — suitable for streaming and online learning.
- **Trivial parallelism** — the hash is stateless.

## Trade-off

**Collisions** distort individual features but do not bias expected
inner products under random hashing. Increase `d` to shrink
collision noise.

## When to use

- **Text n-gram bags** with millions of possible tokens.
- **High-cardinality categoricals** (URLs, user IDs) in memory-
  constrained pipelines.
- **Online learning** where the feature vocabulary evolves.

## When NOT to use

- **Interpretability required** — hashed features have no readable
  name.
- **Small `d`** — collisions dominate; use target encoding or one-
  hot instead.

## Files

- `python/feature_hashing.py` — signed feature hashing + OLS-RSS
  vs `d` sweep. Demo (500 documents, vocab 100 tokens): OLS RSS
  drops **17 904 (d=4) → 8 397 (16) → 2 363 (64) → 612 (256)** as
  `d` grows.
- `r/feature_hashing.R` — `FeatureHashing::hashed.model.matrix`,
  `text2vec::hash_vectorizer` (R); `sklearn.FeatureHasher`,
  `category_encoders.HashingEncoder` (Python).

## Assumptions & caveats

- **Hash function quality** — production tools use MurmurHash3 /
  SipHash; MD5 in this reference is illustrative only.
- **Sign bit** matters — unsigned hashing can bias linear models.
- **Feature importance** on hashed features is uninterpretable; use
  the hash as a black-box featureizer.
- **Different random hashes** across train / test destroy the
  encoding — pin the hash function and seed.

## Related in this repo

- `target-encoding`, `dummy-contrast-coding` — alternative
  categorical encodings.
- `random-projections` — the dense-numeric analogue.

## Run

```
python techniques/feature-hashing/python/feature_hashing.py
Rscript techniques/feature-hashing/r/feature_hashing.R
```

**Refs:** Weinberger, K., Dasgupta, A., Langford, J., Smola, A., & Attenberg, J. "Feature hashing for large scale multitask learning." *ICML*, 2009.

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
