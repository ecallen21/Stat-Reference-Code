# HSIC kernel independence test (Reference Sec 47.56)
# Native R via dHSIC / energy; Python via hyppo / from-scratch.
# Run with:  Rscript hsic_independence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dHSIC::dhsic.test         -- d-variable HSIC test with permutation p\n")
  cat("  energy::dcov / dcor       -- distance covariance / correlation cousin\n")
  cat("  kernlab::mmd              -- MMD two-sample test in the same framework\n")
  cat("Python:\n")
  cat("  hyppo.independence.Hsic   -- HSIC with permutation / gamma p-value\n")
  cat("  scikit-multiflow          -- streaming HSIC drift detection\n")
  cat("  from-scratch              -- see hsic_independence.py\n")
  cat("Refs: Gretton, Bousquet, Smola & Scholkopf (2005) ALT; Gretton et al (2008)\n")
  cat("      'A kernel statistical test of independence', NeurIPS.\n")
}
