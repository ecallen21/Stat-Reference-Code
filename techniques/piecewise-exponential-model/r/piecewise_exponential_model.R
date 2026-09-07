# Piecewise exponential model (Reference Sec 11.26)
# Native R via survival::pyears + glm; Python via lifelines.
# Run with:  Rscript piecewise_exponential_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::pyears + stats::glm(family = poisson)  -- split + Poisson MLE\n")
  cat("  eha::phreg / weibreg / aftreg\n")
  cat("  pch                                              -- dedicated piecewise-hazard\n")
  cat("  rstpm2 / flexsurv                                 -- flexible parametric\n")
  cat("Python:\n")
  cat("  lifelines.PiecewiseExponentialRegressionFitter\n")
  cat("  statsmodels + person-time expansion + Poisson GLM (manual)\n")
  cat("Refs: Friedman, M. (1982) 'Piecewise exponential models for survival data\n")
  cat("      with covariates', Ann Stat 10(1): 101-113; Holford, T.R. (1980)\n")
  cat("      'The analysis of rates and of survivorship using log-linear models',\n")
  cat("      Biometrics 36(2): 299-305.\n")
}
