# Meta-regression (Reference Sec 22.5)
# Native R via metafor::rma; Python custom.
# Run with:  Rscript meta_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::rma / rma.mv             -- REML meta-analysis + meta-regression\n")
  cat("  meta::metareg                     -- moderator regression\n")
  cat("  brms                              -- Bayesian meta-regression\n")
  cat("Python:\n")
  cat("  statsmodels + custom              -- REML meta-regression\n")
  cat("  pymare                            -- Python meta-analysis (moderator support)\n")
  cat("Refs: Thompson & Sharp (1999) 'Explaining heterogeneity in meta-analysis: a\n")
  cat("      comparison of methods', Stat Med; Viechtbauer (2010) 'Conducting meta-\n")
  cat("      analyses in R with the metafor package', JSS.\n")
}
