# Structural breaks + interrupted time series (Reference §13.7, §13.10)
# Base R via strucchange::sctest / breakpoints + lm() for ITS.
# Run with:  Rscript structural_breaks_its.R

if (sys.nframe() == 0) {
  set.seed(43); n <- 100; break_t <- 50
  t <- 1:n
  y <- 10 + 0.2 * t + rnorm(n)
  y[(break_t + 1):n] <- y[(break_t + 1):n] + 5 - 0.2 * ((break_t + 1):n - break_t)
  if (requireNamespace("strucchange", quietly = TRUE)) {
    cat("=== strucchange::sctest (Chow) ===\n")
    print(strucchange::sctest(y ~ t, type = "Chow", point = break_t))
    cat("\n=== strucchange::breakpoints (Bai-Perron) ===\n")
    print(strucchange::breakpoints(y ~ t))
  }
  cat("\n=== ITS regression ===\n")
  D <- as.integer(t >= break_t)
  print(summary(lm(y ~ t + D + I((t - break_t) * D))))
}
