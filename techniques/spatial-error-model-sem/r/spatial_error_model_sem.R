# Spatial Error Model (Reference Sec 47.48)
# Native R via spatialreg / spdep; Python via pysal.
# Run with:  Rscript spatial_error_model_sem.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatialreg::errorsarlm    -- ML fit of SEM\n")
  cat("  spatialreg::GMerrorsar    -- GMM-based SEM (Kelejian-Prucha)\n")
  cat("  spdep::lm.LMtests         -- Lagrange multiplier lag-vs-error tests\n")
  cat("Python:\n")
  cat("  pysal.model.spreg.ML_Error / GM_Error  -- ML & GMM SEM\n")
  cat("  pysal.model.spreg.GM_Combo             -- SARAR combo model\n")
  cat("  from-scratch                            -- see spatial_error_model_sem.py\n")
  cat("Refs: Anselin (1988) 'Spatial Econometrics', Kluwer; Kelejian & Prucha\n")
  cat("      (1999) Int Econ Rev 40(2); Elhorst (2010) Spatial Econ Anal 5(1).\n")
}
