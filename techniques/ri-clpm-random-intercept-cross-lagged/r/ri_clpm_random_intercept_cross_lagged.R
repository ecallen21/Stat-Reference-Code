# RI-CLPM -- random-intercept cross-lagged panel (Reference Sec 20.30)
# Native R via lavaan / OpenMx; Python via semopy.
# Run with:  Rscript ri_clpm_random_intercept_cross_lagged.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lavaan::sem  with RI-CLPM model syntax  -- SEM implementation\n")
  cat("  OpenMx::mxRun with RI-CLPM setup        -- explicit-matrix SEM\n")
  cat("  brms::brm with (1|id) random intercepts  -- Bayesian RI-CLPM\n")
  cat("  psychonetrics                            -- graphical panel models\n")
  cat("Python:\n")
  cat("  semopy (Python SEM library)              -- lavaan-style syntax\n")
  cat("  pymc / numpyro (custom hierarchical VAR)\n")
  cat("  statsmodels.MixedLM (per-equation random intercept fallback)\n")
  cat("Refs: Hamaker, E.L., Kuiper, R.M. & Grasman, R.P.P.P. (2015) 'A critique\n")
  cat("      of the cross-lagged panel model', Psych Methods 20(1); Mulder,\n")
  cat("      J.D. & Hamaker, E.L. (2020) 'Three extensions of the random\n")
  cat("      intercept cross-lagged panel model', SEM: A Multi J 28(4).\n")
}
