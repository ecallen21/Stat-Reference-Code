# Quantile IV regression (Chernozhukov-Hansen) (Reference Sec 15.44)
# Native R via quantreg + IVQR; Python via linearmodels + custom.
# Run with:  Rscript quantile_iv_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quantreg::rq              -- non-IV quantile regression\n")
  cat("  ivqte                     -- Frolich-Melly IV quantile treatment\n")
  cat("  qte                        -- Callaway QTE panel / DiD variants\n")
  cat("  IVQuantile                -- Chernozhukov-Hansen implementation\n")
  cat("Python:\n")
  cat("  econml.iv.dr / iv.dml     -- causal IV / DR-QTE\n")
  cat("  From-scratch scipy + quantreg custom (see quantile_iv_regression.py)\n")
  cat("Refs: Chernozhukov, V. & Hansen, C. (2005) 'An IV model of quantile\n")
  cat("      treatment effects', Econometrica 73(1); Chernozhukov & Hansen (2008)\n")
  cat("      'Instrumental variable quantile regression', Econometrics J 11(1).\n")
}
