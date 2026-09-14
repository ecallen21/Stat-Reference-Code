# Expected Calibration Error (Reference Sec 47.283)
# Native R via probably / yardstick; Python via netcal / torchmetrics / from-scratch.
# Run with:  Rscript expected_calibration_error.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  probably::cal_plot_windowed  -- ECE + reliability plots\n")
  cat("  yardstick::brier_class       -- Brier score (related proper score)\n")
  cat("  Metrics::logLoss             -- log score complement\n")
  cat("Python:\n")
  cat("  netcal.metrics.ECE / MCE / ACE\n")
  cat("  torchmetrics.CalibrationError\n")
  cat("  From-scratch numpy (see expected_calibration_error.py)\n")
  cat("Refs: Naeini, M.P., Cooper, G.F. & Hauskrecht, M. (2015) 'Obtaining\n")
  cat("      well calibrated probabilities using Bayesian binning', AAAI;\n")
  cat("      Guo, C. et al (2017) 'On calibration of modern neural\n")
  cat("      networks', ICML.\n")
}
