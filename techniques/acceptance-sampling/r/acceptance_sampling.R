# Acceptance sampling (Reference Sec 37.8 / 37.13)
# Native R via AcceptanceSampling; Python custom.
# Run with:  Rscript acceptance_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  AcceptanceSampling            -- OC / AOQ / ATI for single / double plans\n")
  cat("  RcmdrPlugin.qcc               -- GUI wrapper\n")
  cat("  qcc                            -- includes basic AS plan helpers\n")
  cat("Python:\n")
  cat("  scipy.stats.binom + custom loop\n")
  cat("  spc                            -- adjacent SPC toolkit\n")
  cat("Refs: Dodge, H.F. & Romig, H.G. (1959) 'Sampling Inspection Tables',\n")
  cat("      Wiley (LTPD tables); Schilling, E.G. (2009) 'Acceptance Sampling in\n")
  cat("      Quality Control', 2nd ed., CRC.\n")
}
