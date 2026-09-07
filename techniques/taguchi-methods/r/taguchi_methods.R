# Taguchi methods (Reference Sec 17.15)
# Native R via DoE.base / qualityTools; Python pyDOE2 + custom.
# Run with:  Rscript taguchi_methods.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DoE.base::oa.design               -- orthogonal-array generator (L4, L8, ...)\n")
  cat("  qualityTools                       -- Taguchi + SNR + robust design\n")
  cat("  SixSigma::ss.ci                    -- Six Sigma DOE utilities\n")
  cat("Python:\n")
  cat("  pyDOE2                             -- orthogonal arrays, DOE templates\n")
  cat("  custom (numpy + scipy)              -- SNR + level analysis\n")
  cat("Refs: Taguchi, G. (1986) Introduction to Quality Engineering, Asian Productivity\n")
  cat("      Organization; Ross, P.J. (1996) Taguchi Techniques for Quality Engineering,\n")
  cat("      McGraw-Hill.\n")
}
