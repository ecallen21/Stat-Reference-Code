# Sliced Inverse Regression + SAVE (Reference Sec 25.3)
# Native R via dr / edrGraphicalTools; Python via reticulate.
# Run with:  Rscript sir_sufficient_dim_reduction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dr                           -- Weisberg-Cook SIR, SAVE, IHT, pHd\n")
  cat("  edrGraphicalTools            -- SIR, SAVE, edr with graphical diagnostics\n")
  cat("  ldr                           -- likelihood-based dim reduction\n")
  cat("Python:\n")
  cat("  sliced                        -- SIR / SAVE / PHD Python package\n")
  cat("  sklearn (PLS as adjacent)      -- supervised covariance-based DR\n")
  cat("Refs: Li, K.-C. (1991) 'Sliced inverse regression for dimension reduction', JASA;\n")
  cat("      Cook, R.D. & Weisberg, S. (1991) 'Discussion of Sliced Inverse Regression\n")
  cat("      for Dimension Reduction (SAVE)', JASA.\n")
}
