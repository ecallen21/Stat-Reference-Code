# Temperature Scaling (Reference Sec 47.281)
# Native R via probably / custom; Python via netcal / torchcalibration / from-scratch.
# Run with:  Rscript temperature_scaling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  probably::cal_estimate_beta   -- calibration wrappers (Beta, isotonic)\n")
  cat("  probably::cal_estimate_logistic -- Platt-style logistic (analogous)\n")
  cat("  custom optim() on nll(T)       -- 1-D scalar optimisation\n")
  cat("Python:\n")
  cat("  netcal.scaling.TemperatureScaling\n")
  cat("  torchcalibration.TemperatureScaling\n")
  cat("  From-scratch scipy (see temperature_scaling.py)\n")
  cat("Refs: Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017)\n")
  cat("      'On calibration of modern neural networks', ICML.\n")
}
