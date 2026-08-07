# Concordance C-index (Reference §11.6)
# R via survival::concordance or Hmisc::rcorr.cens.
# Run with:  Rscript harrell_c_index.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  x <- rnorm(n); T <- rexp(n, exp(0.7 * x)); C <- rexp(n, 1/3)
  time <- pmin(T, C); event <- as.integer(T <= C)
  pred <- 0.7 * x
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::concordance ===\n")
    print(survival::concordance(survival::Surv(time, event) ~ pred))
  }
  if (requireNamespace("Hmisc", quietly = TRUE)) {
    cat("\n=== Hmisc::rcorr.cens ===\n")
    print(Hmisc::rcorr.cens(pred, survival::Surv(time, event)))
  }
}
