# Event-study design (Reference Sec 35.11)
# Native R via fixest; Python via pyfixest.
# Run with:  Rscript event_study.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fixest::feols(y ~ i(event, ref = -1) | unit + time)   -- OLS event study\n")
  cat("  fixest::sunab                                          -- Sun-Abraham 2021\n")
  cat("  did (Callaway-Sant'Anna)                              -- staggered event study\n")
  cat("Python:\n")
  cat("  pyfixest                       -- feols + i() operator + sunab\n")
  cat("  linearmodels.iv.PanelOLS + custom event dummies\n")
  cat("Refs: Borusyak, K., Jaravel, X. & Spiess, J. (2024) 'Revisiting event study\n")
  cat("      designs: robust and efficient estimation'; Sun, L. & Abraham, S. (2021)\n")
  cat("      'Estimating dynamic treatment effects in event studies with heterogeneous\n")
  cat("      treatment effects', J. Econometrics.\n")
}
