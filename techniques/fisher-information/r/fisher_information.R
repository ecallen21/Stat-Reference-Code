# Fisher information + Cramer-Rao (Reference Sec 34.4)
# Native R via numDeriv; Python via statsmodels.
# Run with:  Rscript fisher_information.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  numDeriv::hessian             -- numeric second-derivative for I(theta)\n")
  cat("  MASS::stepAIC                 -- AIC-based model selection (adjacent)\n")
  cat("  stats::vcov                    -- Fisher-info-based covariance for GLMs\n")
  cat("Python:\n")
  cat("  statsmodels.tools.numdiff.approx_hess   -- numeric Hessian\n")
  cat("  scipy.optimize + explicit Hessian\n")
  cat("  torch.autograd + jax.hessian  -- autodiff of log-likelihood\n")
  cat("Refs: Fisher, R.A. (1922) 'On the mathematical foundations of theoretical\n")
  cat("      statistics', Phil Trans R Soc A; Cramer, H. (1946) 'Mathematical\n")
  cat("      Methods of Statistics', Princeton U.P.; Rao, C.R. (1945).\n")
}
