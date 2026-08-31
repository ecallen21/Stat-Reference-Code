# Functional regression (Reference Sec 31.3)
# Native R via fda / refund; Python via scikit-fda.
# Run with:  Rscript functional_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda::fRegress                -- Ramsay-Silverman scalar-on-function\n")
  cat("  refund::pfr                  -- Goldsmith et al. penalised functional regression\n")
  cat("  fda.usc                      -- functional lm, kernel functional regression\n")
  cat("  FDboost                      -- boosted functional regression\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- LinearScalarRegression, functional GLMs\n")
  cat("  pyfda                          -- basic FDA + regression\n")
  cat("Refs: Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis',\n")
  cat("      Springer, Ch. 12-13; Goldsmith, J. et al. (2011) 'Penalized functional\n")
  cat("      regression', J Comp Graph Stat.\n")
}
