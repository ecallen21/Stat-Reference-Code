# Empirical likelihood (Reference Sec 33.5)
# Native R via emplik; Python via reticulate.
# Run with:  Rscript empirical_likelihood.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  emplik                       -- Owen's EL ratio tests + CIs; Kim-Lin extensions\n")
  cat("  survELtest, ELYP             -- EL for survival and quantile regression\n")
  cat("  gmm                          -- generalized method of moments (EL-adjacent)\n")
  cat("Python:\n")
  cat("  empirical-likelihood         -- pip package with mean / regression EL\n")
  cat("  statsmodels.emplike          -- built-in EL for means and regression\n")
  cat("Refs: Owen, A.B. (1988) 'Empirical likelihood ratio confidence intervals for a\n")
  cat("      single functional', Biometrika; Owen, A.B. (2001) 'Empirical Likelihood', CRC.\n")
}
