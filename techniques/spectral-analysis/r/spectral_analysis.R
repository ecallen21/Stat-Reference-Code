# Spectral analysis (Reference §13.18)
# R via stats::spectrum (raw + Daniell-smoothed periodogram).
# Run with:  Rscript spectral_analysis.R

if (sys.nframe() == 0) {
  set.seed(1); T_ <- 512
  t <- 0:(T_ - 1)
  y <- 1.5 * sin(2 * pi * t / 20) + 0.7 * sin(2 * pi * t / 8) + rnorm(T_, 0, 0.5)

  cat("=== stats::spectrum (raw periodogram) ===\n")
  sp_raw <- spectrum(y, plot = FALSE, taper = 0)
  idx <- which.max(sp_raw$spec)
  cat(sprintf("  peak freq = %.4f  ->  period = %.2f\n",
              sp_raw$freq[idx], 1 / sp_raw$freq[idx]))

  cat("\n=== stats::spectrum (Daniell-smoothed, span = 7) ===\n")
  sp_sm <- spectrum(y, spans = 7, plot = FALSE)
  idx <- which.max(sp_sm$spec)
  cat(sprintf("  peak freq = %.4f  ->  period = %.2f\n",
              sp_sm$freq[idx], 1 / sp_sm$freq[idx]))
}
