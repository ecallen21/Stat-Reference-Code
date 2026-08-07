# Dynamic Time Warping (Reference §13.22)
# R via dtw::dtw or dtwclust::dtw_basic.
# Run with:  Rscript dynamic_time_warping.R

if (sys.nframe() == 0) {
  set.seed(0); N <- 100
  t <- seq(0, 4 * pi, length.out = N)
  x <- sin(t); y <- sin(t + 0.6)
  euc <- sqrt(sum((x - y)^2))
  cat(sprintf("Euclidean distance: %.3f\n", euc))
  if (requireNamespace("dtw", quietly = TRUE)) {
    d <- dtw::dtw(x, y, keep = TRUE)
    cat(sprintf("DTW distance:       %.3f\n", d$distance))
    cat(sprintf("DTW normalized:     %.3f\n", d$normalizedDistance))
  }
}
