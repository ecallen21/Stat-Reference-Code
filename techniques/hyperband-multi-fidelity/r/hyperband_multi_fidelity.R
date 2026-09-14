# Hyperband (Reference Sec 47.275)
# Native R via mlr3tuning / mlr3hyperband; Python via Ray Tune / Optuna / from-scratch.
# Run with:  Rscript hyperband_multi_fidelity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlr3hyperband              -- Hyperband tuner for mlr3\n")
  cat("  mlr3tuning + mlr3learners  -- generic tuning framework wrappers\n")
  cat("  reticulate::use_python + Ray Tune\n")
  cat("Python:\n")
  cat("  Ray Tune HyperBandScheduler\n")
  cat("  Optuna HyperbandPruner\n")
  cat("  hpbandster (HBHB implementations)\n")
  cat("  From-scratch (see hyperband_multi_fidelity.py)\n")
  cat("Refs: Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. & Talwalkar, A.\n")
  cat("      (2018) 'Hyperband: A novel bandit-based approach to hyperparameter\n")
  cat("      optimization', J Mach Learn Res 18(185), 1-52.\n")
}
