# Generalized Method of Moments (Reference Sec 35.5)
# Native R via gmm; Python via statsmodels.
# Run with:  Rscript gmm_general.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gmm                          -- Chausse reference: 1-step, 2-step, CUE, ITER-GMM\n")
  cat("  momentfit                     -- successor covering GEL / CUE variants\n")
  cat("  emplik                        -- adjacent empirical-likelihood method\n")
  cat("Python:\n")
  cat("  statsmodels.sandbox.regression.gmm.GMM\n")
  cat("  linearmodels.iv.IV2SLS / IVGMM -- IV as GMM special case\n")
  cat("Refs: Hansen, L.P. (1982) 'Large sample properties of generalized method of\n")
  cat("      moments estimators', Econometrica; Hall, A.R. (2005) 'Generalized Method\n")
  cat("      of Moments', Oxford U.P.\n")
}
