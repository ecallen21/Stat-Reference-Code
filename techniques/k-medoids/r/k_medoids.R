# k-medoids / PAM (Reference §9.10)
# R via cluster::pam (authoritative).
# Run with:  Rscript k_medoids.R

if (sys.nframe() == 0) {
  set.seed(3)
  centers <- rbind(c(0, 0), c(4, 4), c(8, 0))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i)
      MASS::mvrnorm(40, centers[i, ], diag(2) * 0.49)))
  X <- rbind(X, c(20, 20))                # outlier
  if (requireNamespace("cluster", quietly = TRUE)) {
    cat("=== cluster::pam (k = 3) ===\n")
    fit <- cluster::pam(X, k = 3)
    print(fit)
  }
}
