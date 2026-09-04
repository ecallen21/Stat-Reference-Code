# Measurement error models (Reference Sec 38.7)
# Native R via simex / mecor; Python custom + scipy.odr.
# Run with:  Rscript measurement_error_models.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  simex                          -- Cook-Stefanski SIMEX\n")
  cat("  mecor                          -- regression calibration, MI, VC\n")
  cat("  merror, eivtools               -- specialised toolboxes\n")
  cat("Python:\n")
  cat("  scipy.odr                      -- orthogonal distance regression\n")
  cat("  custom                         -- SIMEX + regression calibration\n")
  cat("Refs: Carroll, Ruppert, Stefanski, & Crainiceanu (2006) Measurement Error in\n")
  cat("      Nonlinear Models, 2nd ed., Chapman & Hall; Fuller, W.A. (1987) Measurement\n")
  cat("      Error Models, Wiley; Cook & Stefanski (1994) SIMEX, JASA.\n")
}
