# Six Sigma statistical methods (Reference Sec 37.9)
# Native R via SixSigma; Python via custom.
# Run with:  Rscript six_sigma_methods.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SixSigma                     -- Cano Six Sigma reference package\n")
  cat("  qcc                            -- adjacent SPC + capability\n")
  cat("Python:\n")
  cat("  sixsigma (pip pkg)             -- DPMO / sigma / DMAIC helpers\n")
  cat("  matplotlib + custom            -- manual\n")
  cat("Refs: Motorola Inc. (1986) 'Six Sigma quality initiative'; Pyzdek, T. &\n")
  cat("      Keller, P. (2018) 'The Six Sigma Handbook', 5th ed., McGraw-Hill.\n")
}
