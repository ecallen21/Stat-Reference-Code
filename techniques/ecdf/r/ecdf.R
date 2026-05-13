# Empirical Cumulative Distribution Function (Reference §1.13)
# From-scratch base-R plus stats::ecdf. Run with:  Rscript ecdf.R
#
# Inputs used below:
#   data  : numeric vector (the sample F_n is built from)
#   Fn    : the closure returned by ecdf_scratch (carries the sorted data via attributes)
#   t     : value(s) at which to evaluate F_n
#   p     : probability in (0, 1] (for the inverse ECDF / quantile)
#   alpha : 1 - confidence level for the simultaneous DKW band (0.05 -> 95%)

# From scratch -------------------------------------------------------------
ecdf_scratch <- function(data) {
  xs <- sort(as.numeric(data)); n <- length(xs)
  stopifnot(n > 0)
  Fn <- function(t) vapply(t, function(v) sum(xs <= v) / n, numeric(1))
  attr(Fn, "x") <- xs; attr(Fn, "n") <- n
  Fn
}

ecdf_quantile <- function(Fn, p) {           # inverse ECDF (HF type 1)
  xs <- attr(Fn, "x"); n <- attr(Fn, "n")
  if (p == 0) return(xs[1])
  stopifnot(p > 0, p <= 1)
  xs[min(ceiling(p * n), n)]
}

dkw_band <- function(Fn, alpha = 0.05) {       # simultaneous 1-alpha band
  xs <- attr(Fn, "x"); n <- attr(Fn, "n")
  eps <- sqrt(log(2 / alpha) / (2 * n))
  fhat <- Fn(xs)
  data.frame(x = xs, Fn = fhat,
             lower = pmax(0, fhat - eps), upper = pmin(1, fhat + eps))
}

# Library: stats::ecdf returns a step function; knots()/environment give the data
library_demo <- function(data, grid) {
  Fn <- stats::ecdf(data)
  setNames(Fn(grid), paste0("ecdf(", grid, ")"))
}

if (sys.nframe() == 0) {
  data <- c(2, 3, 3, 5, 8, 13, 21); grid <- c(1, 3, 4, 8, 25)
  Fn <- ecdf_scratch(data)
  cat("data:", data, "\n\n--- from scratch ---\n")
  for (t in grid) cat(sprintf("F_n(%2d) = %.4f\n", t, Fn(t)))
  cat("inverse ECDF p=0.5 :", ecdf_quantile(Fn, 0.5), "\n")
  cat("inverse ECDF p=0.9 :", ecdf_quantile(Fn, 0.9), "\n")
  cat("95% DKW band:\n"); print(dkw_band(Fn))
  cat("\n--- library (stats::ecdf) ---\n"); print(library_demo(data, grid))
}
