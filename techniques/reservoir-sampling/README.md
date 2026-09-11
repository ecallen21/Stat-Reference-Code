# Reservoir Sampling (Reference §47.246)

Vitter (1985). Draw a uniform sample of size `k` from a stream
of unknown length in ONE PASS with O(k) memory:

```
Algorithm R:
    keep first k items
    for i = k+1, k+2, ...:
        j ~ Uniform{0, ..., i-1}
        if j < k: reservoir[j] = item i
```

Each item ends up in the reservoir with probability `k / n`
(`n` = total stream length). Weighted variant (Efraimidis &
Spirakis 2006, A-Res) uses keys `U^(1/w)`.

## Files

- `python/reservoir_sampling.py` — Algorithm R + weighted
  A-Res (top-k by key). Demo: N=100 000 stream, k=100. Sample
  mean ≈ 50 000 (theoretical 49 999.5); empirical inclusion
  probability 0.0010 matches k/N = 0.0010 after 100 trials.
  Weighted reservoir with `w_i = (i+1)²` skews the sampled
  mean id ≈ 76 000 as expected.
- `r/reservoir_sampling.R` — `base::sample`, `stream` package,
  custom loop (R); numpy for known-length, from-scratch
  generator (Python).

## When to use

- **Streaming / unbounded input** where the length is not known
  in advance.
- **Single-pass** required (log processing, telemetry, event
  streams).
- **Memory-limited** subsample of a large dataset that does not
  fit in RAM.
- **Weighted sampling** with unknown normaliser (A-Res).

## When NOT to use

- **Length known ahead of time** — plain `numpy.random.choice`
  or `dplyr::slice_sample` is faster and simpler.
- **Non-uniform probabilities not expressible as weights** —
  use rejection sampling instead.
- **Distributed / parallel** streams — combine per-worker
  reservoirs via a merge step (see Merge-Res).

## Assumptions & caveats

- **Independence of items** — the sample is uniform only if the
  stream ordering is treated as a sequence without adversarial
  interaction with the sampler.
- **Numerical care** with `U^(1/w)` when `w` is large or small
  — work in log space.
- **`k` fixed** — no post-hoc resizing without re-visiting the
  stream.
- **Concurrency** — a shared reservoir needs a lock; per-worker
  reservoirs merged with a weighted concatenation.

## Related in this repo

- `count-min-sketch` — sketch-based streaming frequency
  estimator.
- `hyperloglog-cardinality` — sub-linear cardinality of a
  stream.
- `min-hash-lsh` — approximate similarity sketch.
- `bootstrap` — resampling with replacement from a fixed
  dataset.

## Run

```
python techniques/reservoir-sampling/python/reservoir_sampling.py
Rscript techniques/reservoir-sampling/r/reservoir_sampling.R
```

**Refs:** Vitter, J.S. "Random Sampling with a Reservoir." *ACM Trans. Math. Softw.*, 11(1): 37-57, 1985; Efraimidis, P.S. and Spirakis, P.G. "Weighted random sampling with a reservoir." *Inf. Process. Lett.*, 97(5): 181-185, 2006.

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
