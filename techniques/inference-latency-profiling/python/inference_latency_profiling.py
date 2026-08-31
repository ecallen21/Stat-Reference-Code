"""Inference latency profiling (Reference Ch 32 MLOps).

Measure inference latency at deployment:
  * DISTRIBUTION: p50, p90, p95, p99, p99.9 (tail matters).
  * BATCH-SIZE SWEEP: throughput vs latency trade-off.
  * WARM-UP effect: first calls include JIT / cache-fill; discard them.
  * SLO CHECK: is p99 latency within budget?

Tail-latency (Dean & Barroso 2013 'The Tail at Scale') is often the
critical SLO for user-facing systems; a mean latency of 50 ms can hide
a p99 of 500 ms.

Here we simulate a fake model with a computation cost per row + a small
random jitter, sweep batch size, and produce a latency report.
"""
from __future__ import annotations    # stdlib

import time     # perf_counter for wall-clock measurement

import numpy as np    # numerical arrays


def fake_model(X):
    """Toy 'model': matmul with a random weight -> mimics a tiny linear layer."""
    W = np.random.default_rng(0).normal(0, 1, (X.shape[1], 10))
    return X @ W


def profile_latency(inference_fn, X, batch_size=1, n_warmup=10, n_iters=200):
    """Run inference_fn on batches and record per-batch latency (ms)."""
    n = X.shape[0]
    # Warm-up (discarded)
    for _ in range(n_warmup):
        _ = inference_fn(X[:batch_size])
    lat = []
    for _ in range(n_iters):
        idx = np.random.default_rng().integers(0, n, batch_size)
        t0 = time.perf_counter()
        _ = inference_fn(X[idx])
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return {"batch_size": batch_size,
             "p50": float(np.percentile(lat, 50)),
             "p90": float(np.percentile(lat, 90)),
             "p95": float(np.percentile(lat, 95)),
             "p99": float(np.percentile(lat, 99)),
             "p99_9": float(np.percentile(lat, 99.9)),
             "mean": float(lat.mean()),
             "std": float(lat.std()),
             "throughput_rows_per_s": batch_size / (lat.mean() / 1000.0)}


if __name__ == "__main__":
    print("=== Inference latency profiling: percentiles + batch-size sweep ===\n")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10_000, 64))

    print(f"  {'batch':>6}  {'p50 ms':>7}  {'p95 ms':>7}  {'p99 ms':>7}"
          f"  {'p99.9':>7}  {'throughput rows/s':>18}")
    for bs in (1, 8, 32, 128, 512):
        r = profile_latency(fake_model, X, batch_size=bs, n_iters=200)
        print(f"  {bs:>6}  {r['p50']:>7.3f}  {r['p95']:>7.3f}  {r['p99']:>7.3f}"
              f"  {r['p99_9']:>7.3f}  {r['throughput_rows_per_s']:>18.1f}")

    # SLO check: p99 must be under 5 ms for batch size 1.
    r1 = profile_latency(fake_model, X, batch_size=1)
    slo = 5.0
    print(f"\n  SLO check: p99 @ batch=1 = {r1['p99']:.3f} ms  (target < {slo} ms)"
          f"   -> {'PASS' if r1['p99'] < slo else 'FAIL'}\n")
    print("--- library cross-check (torch.profiler; tensorflow profiler; nvidia nsys;"
          " prometheus histograms; grafana p99 dashboards) ---")
