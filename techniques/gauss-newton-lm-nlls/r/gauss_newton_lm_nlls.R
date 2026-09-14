# Gauss-Newton / Levenberg-Marquardt (Reference Sec 47.321)
# Native R via nls / minpack.lm / nls.lm; Python via scipy.optimize.least_squares / from-scratch.
# Run with:  Rscript gauss_newton_lm_nlls.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::nls                    -- nonlinear least squares (Gauss-Newton)\n")
  cat("  minpack.lm::nlsLM             -- Marquardt-Levenberg (robust to bad starts)\n")
  cat("  nls.lm.control                -- damping / max-step control\n")
  cat("Python:\n")
  cat("  scipy.optimize.least_squares(method='lm'|'trf'|'dogbox')\n")
  cat("  lmfit                          -- higher-level wrapper with parameter obj\n")
  cat("  From-scratch (see gauss_newton_lm_nlls.py)\n")
  cat("Refs: Levenberg, K. (1944) 'A method for the solution of certain\n")
  cat("      nonlinear problems in least squares', Quart Appl Math 2;\n")
  cat("      Marquardt, D.W. (1963) 'An algorithm for least-squares estimation\n")
  cat("      of nonlinear parameters', SIAM J 11(2).\n")
}
