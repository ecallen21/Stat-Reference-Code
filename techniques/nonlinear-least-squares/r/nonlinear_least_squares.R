# Nonlinear least squares (Reference Sec 35.6)
# Native R via nls / minpack.lm; Python via scipy.optimize.
# Run with:  Rscript nonlinear_least_squares.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::nls                    -- Gauss-Newton default\n")
  cat("  minpack.lm::nlsLM             -- Levenberg-Marquardt (Elzhov)\n")
  cat("  nlme::nlme                    -- nonlinear mixed effects\n")
  cat("Python:\n")
  cat("  scipy.optimize.curve_fit       -- LM under the hood\n")
  cat("  scipy.optimize.least_squares   -- 'lm' / 'trf' / 'dogbox' methods\n")
  cat("  lmfit                          -- rich fitting + CIs + constraints\n")
  cat("Refs: Levenberg, K. (1944) 'A method for the solution of certain non-linear\n")
  cat("      problems in least squares'; Marquardt, D.W. (1963) 'An algorithm for\n")
  cat("      least-squares estimation of nonlinear parameters', SIAM J.\n")
}
