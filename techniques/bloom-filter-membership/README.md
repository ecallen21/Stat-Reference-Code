# Bloom Filter (Reference §47.72)

Bloom (1970). Probabilistic set-membership structure. For `n`
expected elements and target false-positive rate `p`, optimal

    m = ⌈−n · ln p / (ln 2)²⌉   (bits)
    k = ⌊(m / n) · ln 2⌉         (hash functions)

`contains(x)`:
  - Definitely NOT in the set if any bit is 0 (no false negatives).
  - Otherwise in the set with FP rate `(1 − e^{−kn/m})^k ≈ p`.

Immutable / append-only; can't delete without **counting Bloom**
or **cuckoo** filters.

## Files

- `python/bloom_filter_membership.py` — from-scratch Bloom
  filter with double-hashing (Kirsch & Mitzenmacher 2006). Demo
  (n=10 000):
  - target FP 0.001, m=143 776 bits (17.6 KB), k=10 → empirical FP 0.0005
  - target FP 0.010, m= 95 851 bits (11.7 KB), k= 7 → empirical FP 0.0102
  - target FP 0.050, m= 62 353 bits ( 7.6 KB), k= 4 → empirical FP 0.0497
  - **False negatives = 0** across all runs (guaranteed).
- `r/bloom_filter_membership.R` — `bloomfilter` (R);
  `pybloom`, `pybloomfiltermmap3`, from-scratch (Python).

## When to use

- **Cache prefilter** — check if a key exists before hitting slow
  storage (LevelDB, RocksDB use Bloom filters).
- **Duplicate detection in streams** — URL crawlers, log dedup.
- **Blockchain / Merkle SPV clients** — light-client transaction
  filtering.
- **Ad exchanges** — user in list membership at query time.
- **Password blocklist checks** — remove obviously breached ones.

## When NOT to use

- **Zero-error required** — use a hash set / cuckoo filter with
  serialised deletions.
- **Set intersection / difference / union** — Bloom filter's
  approximate operations lose precision quickly.
- **Element listing** — Bloom filter can't enumerate members.
- **Highly skewed insertions** — cuckoo filter better memory
  efficiency near capacity.

## Assumptions & caveats

- **Hash quality** — cryptographic hashes (SHA/BLAKE) plus double
  hashing is a common trick.
- **Growth beyond n** — FP rate degrades quadratically; use
  scalable Bloom filter for unbounded streams.
- **No deletion** — counting Bloom filter adds k-bit counters per
  slot (heavier memory).
- **Optimal k = (m/n) · ln 2** — deviations widen FP.

## Related in this repo

- `hyperloglog-cardinality` — cardinality-of-set sibling.
- `min-hash-lsh` — set similarity + LSH.
- `feature-hashing` — hashing trick for feature spaces.
- `record-linkage`, `data-drift-detection` — deduplication /
  monitoring cousins.

## Run

```
python techniques/bloom-filter-membership/python/bloom_filter_membership.py
Rscript techniques/bloom-filter-membership/r/bloom_filter_membership.R
```

**Refs:** Bloom, B.H. "Space/time trade-offs in hash coding with allowable errors." *CACM* 13(7): 422-426, 1970; Kirsch, A. & Mitzenmacher, M. "Less hashing, same performance: Building a better Bloom filter." *ESA*, 2006.

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
