# Bootstrap optimism correction (Reference Sec 39.4, 39.16)
# Native R via rms::validate; Python custom + sklearn.
# Run with:  Rscript bootstrap_optimism_correction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::validate (B = 200-500)    -- bootstrap optimism-corrected AUC + slope\n")
  cat("  rms::calibrate                 -- bootstrap-corrected calibration curve\n")
  cat("  caret::train (bootstrap)       -- alternate CV-based estimation\n")
  cat("  boot                           -- generic bootstrap toolbox\n")
  cat("Python:\n")
  cat("  sklearn.utils.resample         -- bootstrap generator\n")
  cat("  custom                         -- Efron-Harrell loop from scratch\n")
  cat("Refs: Efron, B. (1983) JASA; Harrell, F.E. (2015) Regression Modeling Strategies\n")
  cat("      2nd ed., Springer, Ch 5; Steyerberg, E.W. (2019) Clinical Prediction Models\n")
  cat("      2nd ed., Springer, Ch 5 & 17.\n")
}
