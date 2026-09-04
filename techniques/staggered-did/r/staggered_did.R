# Staggered Difference-in-Differences (Reference Sec 35.19)
# Native R via did (Callaway-Sant'Anna); Python via differences.
# Run with:  Rscript staggered_did.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  did                          -- Callaway-Sant'Anna 2021 reference (att_gt, aggte)\n")
  cat("  fixest::sunab                -- Sun-Abraham 2021 estimator\n")
  cat("  bacondecomp                   -- Goodman-Bacon TWFE decomposition\n")
  cat("  DIDmultiplegt                 -- de Chaisemartin-D'Haultfoeuille\n")
  cat("Python:\n")
  cat("  differences                   -- port of the R 'did' package\n")
  cat("  pyfixest                      -- Sun-Abraham + interaction-weighted estimators\n")
  cat("Refs: Callaway, B. & Sant'Anna, P.H.C. (2021) 'Difference-in-differences with\n")
  cat("      multiple time periods', Journal of Econometrics;\n")
  cat("      Goodman-Bacon, A. (2021) 'Difference-in-differences with variation in\n")
  cat("      treatment timing', J. Econometrics;\n")
  cat("      Sun, L. & Abraham, S. (2021) 'Estimating dynamic treatment effects in\n")
  cat("      event studies with heterogeneous treatment effects', J. Econometrics.\n")
}
