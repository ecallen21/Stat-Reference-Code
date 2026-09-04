# Circular statistics (Reference Sec 38.3)
# Native R via circular; Python pycircstat + custom.
# Run with:  Rscript circular_statistics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  circular::mle.vonmises, rayleigh.test, watson.test\n")
  cat("  movMF                          -- mixtures of von Mises-Fisher\n")
  cat("  Directional                    -- spherical + directional inference\n")
  cat("Python:\n")
  cat("  pycircstat                     -- circular tests\n")
  cat("  scipy.stats.circmean/circvar/circstd\n")
  cat("Refs: Mardia, K.V. & Jupp, P.E. (2000) Directional Statistics, Wiley;\n")
  cat("      Fisher, N.I. (1993) Statistical Analysis of Circular Data, CUP.\n")
}
