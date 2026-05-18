# Measures of dispersion / spread (Reference §1.2)
# From-scratch base-R implementations plus the idiomatic base/stats calls.
# Run with:  Rscript dispersion.R
#
# Inputs used below:
#   x      : numeric vector (the sample)
#   ddof   : "delta degrees of freedom" -- divisor is n - ddof
#            (ddof = 1 for the sample variance, the default; ddof = 0 for the population)
#   p      : probability in [0, 1] (for quantile_type7)
#   center : "mean" or "median" (which center to use for mean_abs_deviation)
#   scale  : MAD multiplier; 1.4826 gives a consistent estimate of sigma at the normal

value_range <- function(x) max(x) - min(x)

variance_scratch <- function(x, ddof = 1) {
  n <- length(x); stopifnot(n - ddof > 0)
  m <- mean(x); sum((x - m)^2) / (n - ddof)
}

sd_scratch <- function(x, ddof = 1) sqrt(variance_scratch(x, ddof))

quantile_type7 <- function(x, p) {
  s <- sort(x); n <- length(s)
  if (n == 1) return(s[1])
  h <- (n - 1) * p; lo <- floor(h); frac <- h - lo
  hi <- min(lo + 1, n - 1)
  s[lo + 1] + frac * (s[hi + 1] - s[lo + 1])
}

iqr_scratch <- function(x) quantile_type7(x, 0.75) - quantile_type7(x, 0.25)

cv_scratch <- function(x, ddof = 1) {
  m <- mean(x); stopifnot(m != 0); sd_scratch(x, ddof) / m
}

mean_abs_deviation <- function(x, center = c("mean", "median")) {
  center <- match.arg(center)
  c0 <- if (center == "mean") mean(x) else median(x)
  mean(abs(x - c0))
}

median_abs_deviation_scratch <- function(x, scale = 1.4826) {
  med <- median(x); median(abs(x - med)) * scale
}

# Library equivalents: var, sd, range/diff, IQR, stats::mad
library_versions <- function(x) list(
  `range (base)`        = diff(range(x)),
  `var (base, n-1)`     = var(x),
  `sd (base, n-1)`      = sd(x),
  `IQR (base)`          = IQR(x),
  `mad scaled (stats)`  = mad(x),                 # default constant = 1.4826
  `CV (sd/mean)`        = sd(x) / mean(x)
)

if (sys.nframe() == 0) {
  data <- c(4, 8, 6, 5, 3, 9, 7, 11, 6, 100)
  cat("data:", data, "\n\n--- from scratch ---\n")
  cat("range            :", value_range(data), "\n")
  cat("variance (n-1)   :", variance_scratch(data), "\n")
  cat("std dev  (n-1)   :", sd_scratch(data), "\n")
  cat("IQR              :", iqr_scratch(data), "\n")
  cat("CV               :", cv_scratch(data), "\n")
  cat("MAD about mean   :", mean_abs_deviation(data, "mean"), "\n")
  cat("MAD about median :", mean_abs_deviation(data, "median"), "\n")
  cat("median abs dev   :", median_abs_deviation_scratch(data), "\n\n--- library ---\n")
  lv <- library_versions(data)
  for (nm in names(lv)) cat(sprintf("%-22s: %s\n", nm, lv[[nm]]))
}
