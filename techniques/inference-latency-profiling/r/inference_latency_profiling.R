# Inference latency profiling (Reference Ch 32 MLOps)
# Native R via microbenchmark / bench; Python via torch.profiler.
# Run with:  Rscript inference_latency_profiling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  microbenchmark              -- nanosecond-precision timing\n")
  cat("  bench                       -- p50/p95/p99 helpers + summary()\n")
  cat("  profvis                     -- interactive flame graphs\n")
  cat("Python:\n")
  cat("  torch.profiler                       -- op-level GPU/CPU profiling\n")
  cat("  tf.profiler / tensorboard             -- TensorFlow profiler\n")
  cat("  nvidia-nsight-systems (nsys)          -- CUDA kernel timing\n")
  cat("  prometheus / grafana p99 dashboards   -- production tail-latency dashboards\n")
  cat("Refs: Dean, J. & Barroso, L. (2013) 'The Tail at Scale', CACM.\n")
}
