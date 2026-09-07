# Indirect inference (Reference Sec 45.8)
# Native R via ii / indirectInference; Python via custom.
# Run with:  Rscript indirect_inference.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  indirectInference          -- Smith and Gouriéroux-style estimators\n")
  cat("  gmm::gmm(..., wmatrix = 'ident')  -- II implementable via GMM\n")
  cat("  yuima                      -- diffusion II via auxiliary GAM\n")
  cat("Python:\n")
  cat("  From-scratch numpy         -- see indirect_inference.py\n")
  cat("  statsmodels + scipy.optimize -- II via any auxiliary sm.OLS / sm.GLM\n")
  cat("Refs: Gouriéroux, C., Monfort, A. & Renault, E. (1993) 'Indirect\n")
  cat("      inference', J Appl Econometrics 8(S1): S85-S118; Smith, A.A.\n")
  cat("      (1993) 'Estimating nonlinear time-series models using simulated\n")
  cat("      vector autoregressions', J Appl Econometrics 8(S1): S63-S84.\n")
}
