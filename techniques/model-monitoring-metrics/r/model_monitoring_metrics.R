# Model monitoring metrics (Reference Ch 32 MLOps)
# Native R for rolling metrics; Python for the full monitoring stack.
# Run with:  Rscript model_monitoring_metrics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc                         -- Shewhart / EWMA / CUSUM control charts\n")
  cat("  DriftR / drifter            -- rolling drift + performance monitoring\n")
  cat("  performance                 -- rolling classification metrics\n")
  cat("Python:\n")
  cat("  evidently                   -- ready-made monitoring dashboards\n")
  cat("  whylogs                     -- profile-based monitoring + drift + performance\n")
  cat("  arize / fiddler / gantry    -- hosted ML observability platforms\n")
  cat("  seldon-alibi-detect          -- online monitoring with sequential guarantees\n")
  cat("Refs: Shewhart, W. (1931) 'Economic Control of Quality of Manufactured Product',\n")
  cat("      Van Nostrand. Roberts, S.W. (1959) 'Control-chart tests based on\n")
  cat("      geometric moving averages (EWMA)', Technometrics.\n")
}
