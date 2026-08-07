# Wavelet analysis (Reference §13.19)
# R via wavelets::dwt / wavethresh::wd.
# Run with:  Rscript wavelet_analysis.R

if (sys.nframe() == 0) {
  set.seed(0); N <- 512
  t <- seq(0, 4 * pi, length.out = N)
  x_clean <- sin(t) + as.numeric(t > 2 * pi)
  x <- x_clean + rnorm(N, 0, 0.3)

  if (requireNamespace("wavelets", quietly = TRUE)) {
    cat("=== wavelets::dwt (Haar, 3 levels) ===\n")
    w <- wavelets::dwt(x, filter = "haar", n.levels = 3)
    print(sapply(w@W, length))
  }

  if (requireNamespace("wavethresh", quietly = TRUE)) {
    cat("\n=== wavethresh::wd + threshold denoise ===\n")
    wt <- wavethresh::wd(x, filter.number = 1, family = "DaubExPhase")
    wt_thr <- wavethresh::threshold(wt, policy = "universal", type = "soft")
    x_dn <- wavethresh::wr(wt_thr)
    cat(sprintf("  MSE noisy    -> clean = %.4f\n", mean((x - x_clean)^2)))
    cat(sprintf("  MSE denoised -> clean = %.4f\n", mean((x_dn - x_clean)^2)))
  }
}
