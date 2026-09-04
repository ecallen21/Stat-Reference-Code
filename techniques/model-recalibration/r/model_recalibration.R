# Model recalibration (Reference Sec 39.6)
# Native R via rms + predtools; Python custom + sklearn.
# Run with:  Rscript model_recalibration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::val.prob                  -- calibration slope + intercept diagnostics\n")
  cat("  predtools::recalibrate         -- intercept + logistic recalibration\n")
  cat("  pmcalibration                  -- calibration + belt confidence intervals\n")
  cat("Python:\n")
  cat("  sklearn.calibration.CalibratedClassifierCV (Platt / isotonic)\n")
  cat("  custom                         -- Steyerberg-style intercept + slope updates\n")
  cat("Refs: Steyerberg, E.W. (2019) Clinical Prediction Models, 2nd ed., Springer, Ch 20;\n")
  cat("      Janssen, K.J.M. et al. (2008) 'Updating methods improved the performance of\n")
  cat("      a clinical prediction model', J Clin Epi.\n")
}
