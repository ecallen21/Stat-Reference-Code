# Structural VAR (Reference Sec 47.306)
# Native R via vars / bvartools / svars; Python via statsmodels / from-scratch.
# Run with:  Rscript structural_var_svar.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  vars::SVAR                -- A / B model SVAR, Blanchard-Quah\n")
  cat("  svars                     -- extensive SVAR identification (sign, IV)\n")
  cat("  bvartools                 -- Bayesian VAR / SVAR\n")
  cat("  urca / tsDyn              -- companion tools\n")
  cat("Python:\n")
  cat("  statsmodels.tsa.vector_ar.svar_model.SVAR\n")
  cat("  linearmodels (VAR/BVAR)\n")
  cat("  From-scratch (see structural_var_svar.py)\n")
  cat("Refs: Sims, C.A. (1980) 'Macroeconomics and reality', Econometrica 48;\n")
  cat("      Blanchard, O.J. & Quah, D. (1989) 'The dynamic effects of\n")
  cat("      aggregate demand and supply disturbances', AER 79(4).\n")
}
