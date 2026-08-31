# Varying-coefficient model (Reference Sec 33.9)
# Native R has strong support; Python via reticulate.
# Run with:  Rscript varying_coefficient_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mgcv::gam(y ~ s(u, by = x))   -- classic P-spline varying coefficients\n")
  cat("  svcm                          -- spatially-varying coefficient models\n")
  cat("  BayesX / R2BayesX             -- Bayesian VCM with MCMC\n")
  cat("  np                             -- kernel-based local VCM\n")
  cat("Python:\n")
  cat("  pyGAM                          -- LinearGAM(s(u, by=x)) VCM\n")
  cat("  statsmodels                    -- manual local WLS\n")
  cat("Refs: Hastie, T. & Tibshirani, R. (1993) 'Varying-coefficient models',\n")
  cat("      J R Stat Soc B; Fan, J. & Zhang, W. (2008) 'Statistical Methods with\n")
  cat("      Varying Coefficient Models', Statistics and Its Interface.\n")
}
