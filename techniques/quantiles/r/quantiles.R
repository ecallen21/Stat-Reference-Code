# Sample quantiles, percentiles and order statistics (Reference §1.5)
# From-scratch base-R plus stats::quantile (which implements all 9 HF types).
# Run with:  Rscript quantiles.R

# Hyndman-Fan types 1, 6, 7 (7 = R's default).
quantile_hf <- function(x, p, kind = 7) {
  stopifnot(p >= 0, p <= 1)
  s <- sort(x); n <- length(s)
  if (n == 1) return(s[1])
  if (kind == 1) {
    if (p == 0) return(s[1])
    j <- ceiling(n * p)
    return(s[max(1, min(j, n))])
  }
  h <- if (kind == 7) (n - 1) * p else if (kind == 6) (n + 1) * p - 1 else stop("kind in {1,6,7}")
  h <- max(0, min(h, n - 1))
  lo <- floor(h); hi <- min(lo + 1, n - 1)
  s[lo + 1] + (h - lo) * (s[hi + 1] - s[lo + 1])
}

five_number_summary <- function(x, kind = 7) c(
  min = min(x), Q1 = quantile_hf(x, .25, kind), median = quantile_hf(x, .5, kind),
  Q3 = quantile_hf(x, .75, kind), max = max(x))

percentile_rank <- function(x, value) {
  n <- length(x)
  100 * (sum(x < value) + 0.5 * sum(x == value)) / n
}

# Library equivalents: stats::quantile(x, probs, type = 1..9), stats::fivenum, base::ecdf
library_versions <- function(x) list(
  `median (type 7)`        = quantile(x, 0.5, type = 7, names = FALSE),
  `p90 type 1 (inv CDF)`   = quantile(x, 0.9, type = 1, names = FALSE),
  `p90 type 6 (Weibull)`   = quantile(x, 0.9, type = 6, names = FALSE),
  `p90 type 7 (default)`   = quantile(x, 0.9, type = 7, names = FALSE),
  `fivenum (Tukey hinges)` = paste(stats::fivenum(x), collapse = ", ")
)

if (sys.nframe() == 0) {
  data <- c(3, 7, 8, 5, 12, 14, 21, 13, 18)
  cat("data:", sort(data), "\n\n--- from scratch ---\n")
  for (k in c(1, 6, 7)) cat(sprintf("p=0.9, HF type %d: %s\n", k, quantile_hf(data, 0.9, k)))
  cat("five-number summary (type 7): "); print(five_number_summary(data))
  cat("percentile rank of 13       :", percentile_rank(data, 13), "\n\n--- library ---\n")
  lv <- library_versions(data)
  for (nm in names(lv)) cat(sprintf("%-26s: %s\n", nm, lv[[nm]]))
}
