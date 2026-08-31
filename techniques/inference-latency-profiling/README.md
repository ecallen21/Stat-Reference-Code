# Inference Latency Profiling (Reference Ch 32 MLOps)

**Measure inference latency at deployment** — the distribution, not just
the mean — and sweep batch size to trade throughput against tail latency.

## What to measure

- **p50 / p90 / p95 / p99 / p99.9** latency in ms. Tail is what users
  experience; a mean of 50 ms can hide a p99 of 500 ms (Dean & Barroso
  2013, "The Tail at Scale").
- **Throughput** (rows/s or requests/s) at each batch size.
- **Warm-up cost** — first calls include JIT compilation and cache
  fill; discard `n_warmup` iterations before measurement.
- **CPU vs GPU** — profile both if the deployment target is uncertain.

## When to use

- **Before every production launch** — the p99 gates SLO commitments.
- **Model refactor / quantisation / distillation** — measure to prove
  the compression paid off.
- **Batching design** — find the batch size that trades acceptable
  latency for max throughput.
- **Post-mortem** on a latency incident — profile against the pre-
  incident baseline.

## When NOT to use

- **Offline batch pipeline** — throughput matters more than tail
  latency; a simple total-runtime measurement is enough.

## Files

- `python/inference_latency_profiling.py` — from-scratch `profile_latency`
  reporting p50/p95/p99/p99.9/mean/std + throughput. Demo on a fake
  matmul "model" swept over batch sizes 1, 8, 32, 128, 512: throughput
  scales from **35 k rows/s at batch=1** to **4 M rows/s at batch=512**
  with p99 ≤ 0.28 ms; SLO check `p99 @ batch=1 < 5 ms` PASSES.
- `r/inference_latency_profiling.R` — `microbenchmark` / `bench` /
  `profvis` (R); `torch.profiler`, `tf.profiler`, `nvidia-nsight-systems`,
  Prometheus/Grafana p99 histograms (Python / production).

## Assumptions & caveats

- **Wall-clock timing** — variability from OS scheduling, GC, page
  faults dominates. Run `n_iters ≥ 200` and report a distribution, not
  a single number.
- **Warm-up matters** — first few calls include compilation; discard
  10-20 iterations for JIT frameworks.
- **Batch size vs latency** — bigger batches amortise per-call
  overhead but hurt tail latency; find the knee.
- **Server-side vs client-side** — measure both; network + serialisation
  dominate for tiny models.
- **Percentile arithmetic** — never average p99s across time windows
  (they don't add); aggregate raw samples or use HDR-Histograms.
- **Cold vs warm cache** — production p99 usually corresponds to cold-
  cache paths; synthetic benchmarks over-optimistic.

## Related in this repo

- `model-monitoring-metrics` — latency is one of the monitored metrics.
- `canary-deployment` — canary stages gate on p99 latency.
- `quantization-pruning`, `knowledge-distillation` — techniques that
  improve latency.
- `feature-store` — online lookup latency is a big component in
  end-to-end inference latency.

## Run

```
python techniques/inference-latency-profiling/python/inference_latency_profiling.py
Rscript techniques/inference-latency-profiling/r/inference_latency_profiling.R
```

**Refs:** Dean, J. & Barroso, L.A. "The tail at scale." *CACM*, 2013; HdrHistogram library (Gil Tene 2013 for zero-loss latency percentile aggregation).

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
