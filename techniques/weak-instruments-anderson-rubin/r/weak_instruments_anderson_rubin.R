# Weak-instrument robust inference (Reference Sec 15.40)
# Native R via ivmodel / AER; Python via linearmodels.IVGMM.
# Run with:  Rscript weak_instruments_anderson_rubin.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ivmodel::ivmodel               -- AR, CLR, LM, K tests + Fieller CIs\n")
  cat("  AER::ivreg + summary(diagnostics=TRUE)  -- weak-IV F, Wu-Hausman, Sargan\n")
  cat("  ivreg / lmtest                 -- alt diagnostics\n")
  cat("  weakIV                          -- Kleibergen-Paap rank F\n")
  cat("Python:\n")
  cat("  linearmodels.iv.IV2SLS / IVGMM  -- has .first_stage.diagnostics\n")
  cat("  linearmodels.iv.absorbing.AbsorbingLS + anderson-rubin extension\n")
  cat("Refs: Anderson & Rubin (1949) 'Estimation of the parameters of a single\n")
  cat("      equation', Ann Math Stat 20; Staiger & Stock (1997) 'Instrumental\n")
  cat("      variables regression with weak instruments', Econometrica 65(3);\n")
  cat("      Moreira (2003) 'A conditional likelihood ratio test for structural\n")
  cat("      models', Econometrica 71(4).\n")
}
