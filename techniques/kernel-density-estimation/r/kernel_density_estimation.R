# Kernel Density Estimation (Reference §6.21)
# From-scratch base R plus stats::density.
# Run with:  Rscript kernel_density_estimation.R
#
# Inputs:  x (numeric vector), grid (where to evaluate), kernel, h

K_gauss <- function(u) exp(-0.5 * u^2) / sqrt(2 * pi)
K_epan  <- function(u) ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0)
K_unif  <- function(u) ifelse(abs(u) <= 1, 0.5, 0)
K_tri   <- function(u) ifelse(abs(u) <= 1, 1 - abs(u), 0)

KERNELS <- list(gaussian = K_gauss, epanechnikov = K_epan,
                uniform = K_unif, triangular = K_tri)

silverman_h <- function(x) 0.9 * min(sd(x), IQR(x) / 1.34) * length(x)^(-0.2)
scott_h     <- function(x) sd(x) * length(x)^(-0.2)

kde_evaluate_scratch <- function(x, grid, h = NULL, kernel = "gaussian") {
  if (is.null(h)) h <- silverman_h(x)
  Kfn <- KERNELS[[kernel]]
  out <- sapply(grid, function(g) sum(Kfn((g - x) / h)) / (length(x) * h))
  list(f = out, h = h)
}

# Library: stats::density (Gaussian kernel by default; "epanechnikov", etc. via kernel=)
library_demo <- function(x) {
  d <- density(x); cat("density() with Silverman h:\n")
  cat("  bw =", d$bw, "  peak at x =", d$x[which.max(d$y)], "\n")
}

if (sys.nframe() == 0) {
  set.seed(8)
  x <- c(rnorm(200, -2, 0.8), rnorm(300, 2, 1.0))
  grid <- seq(-6, 6, length.out = 200)
  cat("Silverman h =", silverman_h(x), "  Scott h =", scott_h(x), "\n")
  for (k in c("gaussian", "epanechnikov", "triangular")) {
    res <- kde_evaluate_scratch(x, grid, kernel = k)
    cat(sprintf("  %-14s: peak at %.2f, area = %.4f, h = %.4f\n",
                k, grid[which.max(res$f)],
                sum(diff(grid) * (head(res$f, -1) + tail(res$f, -1)) / 2),
                res$h))
  }
  cat("\n--- library ---\n"); library_demo(x)
}
