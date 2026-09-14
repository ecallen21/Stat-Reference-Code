# Impulse Response Function (Reference Sec 47.307)
# Native R via vars / svars / lpirfs; Python via statsmodels / from-scratch.
# Run with:  Rscript impulse_response_function.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  vars::irf                   -- standard IRF with bootstrap CIs\n")
  cat("  svars::irf                  -- SVAR-flavoured IRF\n")
  cat("  lpirfs                      -- local-projection IRFs (Jorda 2005)\n")
  cat("Python:\n")
  cat("  statsmodels.tsa.vector_ar.var_model.VARResults.irf\n")
  cat("  linearmodels.iv (Jorda local projections)\n")
  cat("  From-scratch numpy (see impulse_response_function.py)\n")
  cat("Refs: Sims, C.A. (1980) 'Macroeconomics and reality', Econometrica 48;\n")
  cat("      Jorda, O. (2005) 'Estimation and inference of impulse responses\n")
  cat("      by local projections', AER 95(1).\n")
}
