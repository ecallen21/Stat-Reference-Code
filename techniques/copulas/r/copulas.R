# Copulas (Reference Sec 38.9)
# Native R via copula / VineCopula; Python copulas + custom.
# Run with:  Rscript copulas.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  copula (normalCopula, claytonCopula, gumbelCopula, fitCopula)\n")
  cat("  VineCopula                     -- vine (pair) copulas\n")
  cat("  rvinecopulib                   -- fast C++ vine implementation\n")
  cat("Python:\n")
  cat("  copulas (GaussianMultivariate, GumbelMultivariate)\n")
  cat("  pyvinecopulib                  -- vine copulas\n")
  cat("  scipy.stats + custom MLE\n")
  cat("Refs: Nelsen, R.B. (2006) An Introduction to Copulas, 2nd ed., Springer;\n")
  cat("      Joe, H. (2015) Dependence Modeling with Copulas, Chapman & Hall/CRC.\n")
}
