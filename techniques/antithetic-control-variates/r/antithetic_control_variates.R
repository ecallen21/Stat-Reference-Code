# Antithetic variates + control variates (Reference Sec 45.10)
# From-scratch in R and Python; one-liners over runif / rnorm.
# Run with:  Rscript antithetic_control_variates.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- one-liners over base R\n")
  cat("  RQuantLib / fOptions (finance MC with control variates baked in)\n")
  cat("  BASS (Bayesian additive Spline Surfaces) for variance-reduction demos\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see antithetic_control_variates.py)\n")
  cat("  numpy.random + scipy for closed forms; no dedicated package\n")
  cat("Refs: Hammersley, J.M. & Morton, K.W. (1956) 'A new Monte Carlo\n")
  cat("      technique: antithetic variates', Math Proc Cambridge Phil Soc 52;\n")
  cat("      Glasserman (2003) Monte Carlo Methods in Financial Engineering,\n")
  cat("      Springer, chapters 4 and 5.\n")
}
