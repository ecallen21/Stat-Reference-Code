# HyperLogLog Cardinality Estimator (Reference §47.71)

Flajolet, Fusy, Gandouet & Meunier (2007). Estimate the number of
DISTINCT elements in a data stream using O(m) memory,
independent of the input size.

    1. Hash each element to a 64-bit value.
    2. Partition into 2^b registers by leading bits.
    3. Store `rho` = position of leftmost 1 in the remaining bits.
    4. Estimate  E = α_m · m² / Σ 2^{−M[j]}  (with corrections).

Standard error ≈ 1.04 / √m. m = 2¹⁴ → ~0.8 % error in 16 KB.

## Files

- `python/hyperloglog_cardinality.py` — from-scratch HLL with
  SHA-1 hashing + small-range linear-counting fallback. Demo
  across b ∈ {10, 12, 14} and true cardinalities {10³ … 10⁶}:
  - b=10 (4 KB): error 0.1–5 %
  - b=12 (16 KB): error 0.1–3 %
  - b=14 (64 KB): error 0.1–1 %.
- `r/hyperloglog_cardinality.R` — `hll`, `RcppHash` (R);
  `datasketch`, `hyperloglog`, from-scratch (Python).

## When to use

- **Distinct-count over massive streams / logs** — DAUs, unique
  IPs, unique URL visits.
- **SQL-style approximate `COUNT(DISTINCT)`** in analytics
  databases (BigQuery `APPROX_COUNT_DISTINCT`, Redshift, etc).
- **Union of many sub-counts** — HLLs merge by register-wise max,
  enabling MapReduce distinct-count.
- **Cardinality-based Bloom-filter sizing**.

## When NOT to use

- **Exact counts required** — legal, billing, audit contexts.
- **Very small cardinalities** — HLL bias corrected by linear
  counting; a plain hash set is smaller for < 100.
- **Set-similarity / membership queries** — use MinHash or Bloom
  filter instead.

## Assumptions & caveats

- **Uniform hash** — quality of hash function drives quality;
  MurmurHash3 / xxHash standard.
- **Bias** at small cardinality — small-range correction
  (linear counting) or HLL++ (Heule et al 2013).
- **Range extension** — HLL++ handles up to ~10¹⁰ cardinalities
  without large-range bias.
- **Merge is exact under max-register semantics** — enables
  distributed aggregation.

## Related in this repo

- `bloom-filter-membership` — same family (probabilistic data
  structures).
- `min-hash-lsh` — set similarity + near-neighbour retrieval.
- `feature-hashing`, `count-min-sketch` (see `feature-hashing`) —
  sketching cousins.
- `subsampling`, `random-projections` — approximate summary
  techniques.

## Run

```
python techniques/hyperloglog-cardinality/python/hyperloglog_cardinality.py
Rscript techniques/hyperloglog-cardinality/r/hyperloglog_cardinality.R
```

**Refs:** Flajolet, P., Fusy, É., Gandouet, O. & Meunier, F. "HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm." *AofA*, 2007; Heule, S., Nunkesser, M. & Hall, A. "HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm." *EDBT*, 2013.

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
