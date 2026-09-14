# BOHB - Bayesian Optimisation + Hyperband (Reference Sec 47.280)
# Native R via reticulate + Ray Tune; Python via hpbandster / Ray Tune / from-scratch.
# Run with:  Rscript bohb_bayesian_hyperband.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + Ray Tune BOHBScheduler\n")
  cat("  mlr3hyperband + mlr3mbo (approximation via combined tuners)\n")
  cat("Python:\n")
  cat("  hpbandster BOHB               -- authoritative implementation\n")
  cat("  Ray Tune BOHBScheduler        -- distributed variant\n")
  cat("  SMAC3 (Bayesian + successive halving hybrid)\n")
  cat("  From-scratch (see bohb_bayesian_hyperband.py)\n")
  cat("Refs: Falkner, S., Klein, A. & Hutter, F. (2018) 'BOHB: Robust and\n")
  cat("      efficient hyperparameter optimization at scale', ICML.\n")
}
