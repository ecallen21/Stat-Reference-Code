# Tolerance intervals (Reference Sec 38.16)
# Native R via tolerance; Python custom.
# Run with:  Rscript tolerance_intervals.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tolerance                      -- normtol.int, nptol.int, exttol.int, etc.\n")
  cat("  EnvStats::tolIntNorm           -- alternative implementation\n")
  cat("Python:\n")
  cat("  custom                         -- Howe 1969 normal + Wilks 1941 nonparametric\n")
  cat("  scipy.stats.norm.interval      -- confidence interval (NOT tolerance!)\n")
  cat("Refs: Krishnamoorthy, K. & Mathew, T. (2009) Statistical Tolerance Regions,\n")
  cat("      Wiley; Howe, W.G. (1969) 'Two-sided tolerance limits for normal\n")
  cat("      populations', JASA; NIST/SEMATECH e-Handbook Sec 7.2.6.\n")
}
