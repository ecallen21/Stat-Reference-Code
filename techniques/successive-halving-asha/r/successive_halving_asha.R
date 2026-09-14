# Successive Halving / ASHA (Reference Sec 47.276)
# Native R via mlr3tuning; Python via Ray Tune / Optuna / from-scratch.
# Run with:  Rscript successive_halving_asha.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlr3hyperband              -- SH bracket + Hyperband\n")
  cat("  mlr3tuning                 -- generic tuner base\n")
  cat("  reticulate + Ray Tune      -- ASHA via Python\n")
  cat("Python:\n")
  cat("  Ray Tune ASHAScheduler     -- async SH, standard for distributed\n")
  cat("  Optuna SuccessiveHalvingPruner\n")
  cat("  From-scratch (see successive_halving_asha.py)\n")
  cat("Refs: Karnin, Z., Koren, T. & Somekh, O. (2013) 'Almost optimal\n")
  cat("      exploration in multi-armed bandits', ICML;  Li, L. et al (2020)\n")
  cat("      'A system for massively parallel hyperparameter tuning', MLSys.\n")
}
