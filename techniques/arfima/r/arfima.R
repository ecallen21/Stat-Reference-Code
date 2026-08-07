# ARFIMA (Reference §13.16)
# R via fracdiff::fracdiff (Whittle-MLE) or forecast::arfima.
# Run with:  Rscript arfima.R

if (sys.nframe() == 0) {
  set.seed(35); T_ <- 2000
  # Simulate ARFIMA(0, 0.35, 0)
  if (requireNamespace("fracdiff", quietly = TRUE)) {
    x <- fracdiff::fracdiff.sim(T_, d = 0.35)$series
    cat("=== fracdiff::fracdiff MLE on ARFIMA(0, 0.35, 0) ===\n")
    fit <- fracdiff::fracdiff(x)
    cat(sprintf("  d_hat = %.4f (true = 0.35)\n", fit$d))
    cat(sprintf("  SE = %.4f\n", fit$stderror.dpq[1]))
  } else {
    cat("fracdiff not installed; skip.\n")
  }
}
