# Cliff's delta (Reference §7.16)
# R via effsize::cliff.delta.
# Run with:  Rscript cliff_delta.R

if (sys.nframe() == 0) {
  set.seed(0)
  x <- rnorm(40, 0.5); y <- rnorm(40, 0)
  if (requireNamespace("effsize", quietly = TRUE)) {
    cat("=== effsize::cliff.delta ===\n")
    print(effsize::cliff.delta(x, y))
  }
}
