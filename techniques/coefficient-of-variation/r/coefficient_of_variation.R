# Coefficient of variation (Reference §1.22 and §1.33)
# From-scratch base-R plus package calls (DescTools, raster, rstatix).
# Run with:  Rscript coefficient_of_variation.R

cv_scratch <- function(x, as_percent = FALSE) {
  m <- mean(x); stopifnot(m != 0)
  v <- sd(x) / m; if (as_percent) v * 100 else v
}

geometric_cv <- function(x, as_percent = FALSE) {
  stopifnot(all(x > 0))
  s_log <- sd(log(x))
  v <- sqrt(exp(s_log^2) - 1); if (as_percent) v * 100 else v
}

within_subject_cv <- function(subjects, as_percent = FALSE) {
  var_w <- vapply(subjects, var, numeric(1))     # one variance per subject
  overall_mean <- mean(unlist(subjects))
  v <- sqrt(mean(var_w)) / overall_mean; if (as_percent) v * 100 else v
}

# McKay's approximate CI for the population CV (valid for CV < ~0.33)
cv_ci_mckay <- function(x, conf = 0.95) {
  n <- length(x); v <- n - 1; k <- cv_scratch(x); a <- 1 - conf
  chi_hi <- qchisq(1 - a / 2, v) / v; chi_lo <- qchisq(a / 2, v) / v
  c(lo = k / sqrt((chi_hi - 1) * k^2 + chi_hi),
    hi = k / sqrt((chi_lo - 1) * k^2 + chi_lo))
}

# Library: DescTools::CoefVar (has CI methods), raster::cv, base sd(x)/mean(x),
#          rstatix::get_summary_stats(..., type = "common") includes 'cv'
library_demo <- function(x) {
  cat("sd/mean (base)         :", sd(x) / mean(x), "\n")
  if (requireNamespace("DescTools", quietly = TRUE))
    cat("DescTools::CoefVar     :", DescTools::CoefVar(x), "\n")
}

if (sys.nframe() == 0) {
  assay_a <- c(98.2, 101.4, 99.7, 100.1, 102.3, 97.9, 100.8)
  assay_b <- c(4.91, 5.07, 4.98, 5.00, 5.12, 4.90, 5.04)
  cat(sprintf("assay A: mean=%.2f sd=%.3f -> CV = %.2f%%\n", mean(assay_a), sd(assay_a), cv_scratch(assay_a, TRUE)))
  cat(sprintf("assay B: mean=%.2f sd=%.3f -> CV = %.2f%%\n\n", mean(assay_b), sd(assay_b), cv_scratch(assay_b, TRUE)))
  cat("geometric CV of assay A :", round(geometric_cv(assay_a, TRUE), 3), "%\n")
  cat("McKay 95% CI for CV(A)  :", round(cv_ci_mckay(assay_a), 4), "\n")
  reps <- list(c(10.1, 10.3, 9.8), c(12.0, 11.7, 12.2), c(9.5, 9.7, 9.4), c(11.2, 11.0, 11.5))
  cat("within-subject CV       :", round(within_subject_cv(reps, TRUE), 3), "%\n\n--- library ---\n")
  library_demo(assay_a)
}
