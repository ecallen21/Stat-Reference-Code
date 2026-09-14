# Beta Calibration (Reference Sec 47.284)
# Native R via probably / betacal; Python via betacal / netcal / from-scratch.
# Run with:  Rscript beta_calibration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  betacal (R port)                   -- 3-parameter beta calibration\n")
  cat("  probably::cal_estimate_beta        -- tidymodels wrapper\n")
  cat("  bcaboot / custom nls               -- alternative fitters\n")
  cat("Python:\n")
  cat("  betacal (PyPI)                     -- reference implementation\n")
  cat("  netcal.scaling.BetaCalibration\n")
  cat("  From-scratch scipy (see beta_calibration.py)\n")
  cat("Refs: Kull, M., Silva Filho, T.M. & Flach, P. (2017) 'Beta calibration:\n")
  cat("      a well-founded and easily implemented improvement on logistic\n")
  cat("      calibration for binary classifiers', AISTATS.\n")
}
