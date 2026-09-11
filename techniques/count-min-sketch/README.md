# Count-Min Sketch (Reference §47.245)

Cormode & Muthukrishnan (2005). Probabilistic frequency
estimator for streams: `k` hash functions index a `k × w`
counter table, and `estimate(x) = min_j table[j, h_j(x) % w]`.

Space `O(k · w)` regardless of stream size; error bounded by
`ε · (total count)` with probability `1 − δ`, using
`w = ⌈e / ε⌉`, `k = ⌈ln 1/δ⌉`. Always UPPER-BIASED (min of
inflated counters ≥ truth).

## Files

- `python/count_min_sketch.py` — 30-line CMS class with
  hash-seeded table. Demo: 100 000-token Zipf stream over 500
  unique tokens. `w=64, k=3` (1.5 KB) over-estimates by
  ~2563%; `w=1024, k=5` (40 KB) achieves 0% mean over-estimate;
  full exact counts would need 500 × 8 = 4 KB here but scales
  linearly in unique tokens whereas CMS is O(kw) forever.
- `r/count_min_sketch.R` — `bloomfilter`, `digest`-backed custom
  (R); `datasketch.CountMinSketch`, `probables`, from-scratch
  (Python).

## When to use

- **Heavy-hitters on unbounded streams** — top-K query terms,
  IPs, URLs where exact counts do not fit in RAM.
- **Approximate joins / range queries** over streaming data
  (via range-CMS extension).
- **Sub-linear memory** required regardless of universe size.

## When NOT to use

- **Exact counts needed** — CMS only over-estimates; use
  Misra-Gries or an exact hash map.
- **Bounded universe** where a plain array fits — no benefit.
- **Small streams** — the constant overhead outweighs savings.

## Assumptions & caveats

- **Independent hashes** — pairwise-independent hash family
  underpins the `w = e/ε` bound; use MurmurHash / SipHash, not
  Python's process-randomised `hash`.
- **Upper bias grows** with total count and inverse of `w`;
  size `w` for expected max frequency.
- **Deletions (Count-Min-Log, CMSketch with signed counters)**
  extend to increments/decrements but lose the pure upper-bound
  guarantee.
- **Conservative update** (only increment the min-counter arms)
  tightens over-estimation empirically.

## Related in this repo

- `reservoir-sampling` — the other one-pass streaming primitive.
- `hyperloglog-cardinality` — cardinality of a stream in
  O(log log) bits.
- `min-hash-lsh` — approximate similarity search sketch.
- `bloom-filter` — set membership rather than counting.

## Run

```
python techniques/count-min-sketch/python/count_min_sketch.py
Rscript techniques/count-min-sketch/r/count_min_sketch.R
```

**Refs:** Cormode, G. and Muthukrishnan, S. "An Improved Data Stream Summary: The Count-Min Sketch and Its Applications." *Journal of Algorithms*, 55(1): 58-75, 2005.

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
