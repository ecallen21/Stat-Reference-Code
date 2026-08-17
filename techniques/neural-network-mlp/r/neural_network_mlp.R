# Multi-Layer Perceptron (Reference §27.1)
# R via nnet::nnet, keras3, or torch.
# Run with:  Rscript neural_network_mlp.R

if (sys.nframe() == 0) {
  set.seed(0)
  X <- rbind(matrix(rnorm(150 * 2, mean = 0, sd = 0.6), 150, 2),
             matrix(rnorm(150 * 2, mean = 3, sd = 0.6), 150, 2),
             cbind(rnorm(150, 1.5, 0.6), rnorm(150, 3, 0.6)))
  y <- factor(rep(0:2, each = 150))
  if (requireNamespace("nnet", quietly = TRUE)) {
    cat("=== nnet::nnet (1 x 32) ===\n")
    fit <- nnet::nnet(X, class.ind(y), size = 32, softmax = TRUE, maxit = 500, trace = FALSE)
    pred <- max.col(predict(fit, X)) - 1
    cat(sprintf("  accuracy = %.3f\n", mean(pred == as.integer(as.character(y)))))
  }
}
