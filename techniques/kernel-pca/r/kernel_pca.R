# Kernel PCA (Reference §9.10)
# R via kernlab::kpca.
# Run with:  Rscript kernel_pca.R

if (sys.nframe() == 0) {
  set.seed(0); n_per <- 100
  theta_i <- runif(n_per, 0, 2 * pi); r_i <- 1 + rnorm(n_per, 0, 0.1)
  theta_o <- runif(n_per, 0, 2 * pi); r_o <- 3 + rnorm(n_per, 0, 0.1)
  X <- rbind(cbind(r_i * cos(theta_i), r_i * sin(theta_i)),
             cbind(r_o * cos(theta_o), r_o * sin(theta_o)))
  y <- c(rep(0, n_per), rep(1, n_per))
  if (requireNamespace("kernlab", quietly = TRUE)) {
    cat("=== kernlab::kpca (RBF kernel) ===\n")
    fit <- kernlab::kpca(X, kernel = "rbfdot", kpar = list(sigma = 0.5),
                         features = 2)
    Z <- kernlab::rotated(fit)
    cat(sprintf("  class-mean separation on PC1: %.3f\n",
                abs(mean(Z[y == 0, 1]) - mean(Z[y == 1, 1])) / sd(Z[, 1])))
  }
}
