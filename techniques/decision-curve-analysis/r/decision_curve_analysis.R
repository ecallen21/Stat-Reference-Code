# Decision Curve Analysis (Reference §20.x extra)
# R via rmda, dcurves, or DecisionCurve.
# Run with:  Rscript decision_curve_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rmda::decision_curve(y ~ p, data, thresholds=seq(0,1,0.01), bootstraps=500)\n")
  cat("  rmda::plot_decision_curve(list(dc1, dc2))     -- overlay competing models\n")
  cat("  dcurves::dca(y ~ p1 + p2, data)               -- tidymodels-friendly implementation\n")
  cat("  DecisionCurve::dca(...)                        -- Sloan-Kettering original\n")
  cat("Python: dcurves (mirrors R dcurves API).\n")
  cat("Companion: net-benefit integrated over the plausible threshold range (Vickers 2016).\n")
}
