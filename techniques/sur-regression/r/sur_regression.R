# SUR — Seemingly Unrelated Regression (Reference Sec 35.7)
# Native R via systemfit; Python via linearmodels.
# Run with:  Rscript sur_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  systemfit                     -- SUR, 2SLS, 3SLS reference (Henningsen)\n")
  cat("  plm                            -- panel + SUR helpers\n")
  cat("Python:\n")
  cat("  linearmodels.system.SUR       -- Kevin Sheppard's implementation\n")
  cat("  statsmodels.regression.linear_model.OLS + custom FGLS\n")
  cat("Refs: Zellner, A. (1962) 'An efficient method of estimating seemingly unrelated\n")
  cat("      regressions and tests for aggregation bias', JASA;\n")
  cat("      Greene, W.H. (2018) 'Econometric Analysis', 8th ed., Ch. 10.\n")
}
