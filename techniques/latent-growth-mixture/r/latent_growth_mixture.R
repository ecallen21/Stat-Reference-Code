# Latent-growth mixture model (Reference §12.13)
# R via lcmm::hlme (Proust-Lima) or flexmix::flexmix.
# Run with:  Rscript latent_growth_mixture.R

if (sys.nframe() == 0) {
  set.seed(0); N <- 300; T <- 6
  subject <- rep(1:N, each = T); time <- rep(0:(T - 1), N)
  class_true <- sample(0:2, N, replace = TRUE, prob = c(0.4, 0.4, 0.2))
  alpha <- c(1, 5, 3); beta <- c(0.6, -0.4, 0)
  y <- numeric(N * T)
  for (i in 1:N) {
    y[subject == i] <- alpha[class_true[i] + 1] + beta[class_true[i] + 1] * (0:(T - 1)) + rnorm(T, 0, 0.3)
  }
  df <- data.frame(id = subject, time = time, y = y)
  if (requireNamespace("lcmm", quietly = TRUE)) {
    cat("=== lcmm::hlme (K = 3) ===\n")
    fit <- lcmm::hlme(y ~ time, mixture = ~ time, random = ~ 1,
                       subject = "id", ng = 3, data = df)
    print(summary(fit))
  }
}
