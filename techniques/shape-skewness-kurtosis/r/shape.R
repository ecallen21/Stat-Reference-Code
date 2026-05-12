# Measures of distribution shape: skewness and kurtosis (Reference §1.4)
# From-scratch base-R plus package calls (moments, e1071, psych).
# Run with:  Rscript shape.R

central_moment <- function(x, k) { m <- mean(x); mean((x - m)^k) }

skewness_scratch <- function(x, bias = TRUE) {
  n <- length(x)
  g1 <- central_moment(x, 3) / central_moment(x, 2)^1.5
  if (bias) g1 else g1 * sqrt(n * (n - 1)) / (n - 2)
}

kurtosis_scratch <- function(x, bias = TRUE, excess = TRUE) {
  n <- length(x)
  g2 <- central_moment(x, 4) / central_moment(x, 2)^2 - 3
  val <- if (bias) g2 else ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
  if (excess) val else val + 3
}

pearson_second_skewness <- function(x) 3 * (mean(x) - median(x)) / sd(x)

# Library equivalents:
#   moments::skewness / moments::kurtosis  (kurtosis here is NON-excess: normal = 3)
#   e1071::skewness(x, type = 1/2/3), e1071::kurtosis(x, type = ...)
#   psych::describe (returns skew and kurtosis columns)
library_versions <- function(x) {
  out <- list()
  if (requireNamespace("moments", quietly = TRUE)) {
    out$`skewness (moments)` <- moments::skewness(x)
    out$`kurtosis non-excess (moments)` <- moments::kurtosis(x)
  }
  if (requireNamespace("e1071", quietly = TRUE)) {
    out$`skewness type 2 / G1 (e1071)` <- e1071::skewness(x, type = 2)
    out$`excess kurtosis type 2 / G2 (e1071)` <- e1071::kurtosis(x, type = 2)
  }
  if (length(out) == 0) out$note <- "install 'moments' and/or 'e1071' for library versions"
  out
}

if (sys.nframe() == 0) {
  set.seed(0)
  data <- round(rlnorm(200, meanlog = 0, sdlog = 0.6), 3)   # right-skewed
  cat("n =", length(data), " (right-skewed lognormal sample)\n\n--- from scratch ---\n")
  cat("skewness (biased)        :", skewness_scratch(data, TRUE), "\n")
  cat("skewness (G1, corrected) :", skewness_scratch(data, FALSE), "\n")
  cat("excess kurtosis (biased) :", kurtosis_scratch(data, TRUE), "\n")
  cat("excess kurtosis (G2)     :", kurtosis_scratch(data, FALSE), "\n")
  cat("Pearson 2nd skewness     :", pearson_second_skewness(data), "\n\n--- library ---\n")
  lv <- library_versions(data)
  for (nm in names(lv)) cat(sprintf("%-40s: %s\n", nm, lv[[nm]]))
}
