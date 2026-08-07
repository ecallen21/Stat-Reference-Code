# Wald-Wolfowitz Runs Test (Reference §7.15)
# R via tseries::runs.test or randtests::runs.test.
# Run with:  Rscript runs_test.R

if (sys.nframe() == 0) {
  set.seed(0)
  seq <- sample(c(0, 1), 100, replace = TRUE)
  if (requireNamespace("tseries", quietly = TRUE)) {
    cat("=== tseries::runs.test (random binary) ===\n")
    print(tseries::runs.test(as.factor(seq)))
    cat("\n=== tseries::runs.test (alternating 0101...) ===\n")
    print(tseries::runs.test(as.factor(rep(c(0, 1), 50))))
  }
}
