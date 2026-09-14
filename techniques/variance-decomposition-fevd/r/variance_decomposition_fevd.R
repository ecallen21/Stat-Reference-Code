# FEVD (Reference Sec 47.308)
# Native R via vars / svars; Python via statsmodels / from-scratch.
# Run with:  Rscript variance_decomposition_fevd.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  vars::fevd                 -- forecast-error variance decomposition\n")
  cat("  svars::fevd                -- SVAR-identified FEVD\n")
  cat("Python:\n")
  cat("  statsmodels.tsa.vector_ar.var_model.VARResults.fevd\n")
  cat("  From-scratch numpy (see variance_decomposition_fevd.py)\n")
  cat("Refs: Sims, C.A. (1980) 'Macroeconomics and reality', Econometrica 48;\n")
  cat("      Lutkepohl, H. (2005) 'New Introduction to Multiple Time Series\n")
  cat("      Analysis', Springer, ch 2.\n")
}
