# Graph Convolutional Network (Reference Sec 47.148).
#
# Kipf & Welling 2017. Semi-supervised classification via
# symmetric-normalised adjacency propagation.
#
# Two-layer GCN forward pass: illustrative, no training loop.

set.seed(0)
n_per <- 40; K <- 3; n <- n_per * K
y <- rep(seq_len(K) - 1L, each = n_per)
p_in <- 0.15; p_out <- 0.01
A <- matrix(0, n, n)
for (i in seq_len(n - 1)) {
    for (j in (i + 1):n) {
        p <- if (y[i] == y[j]) p_in else p_out
        if (runif(1) < p) { A[i, j] <- 1; A[j, i] <- 1 }
    }
}
A_tilde <- A + diag(n)
d <- rowSums(A_tilde)
D_inv_sqrt <- diag(1 / sqrt(d))
A_hat <- D_inv_sqrt %*% A_tilde %*% D_inv_sqrt

# Random features + random weights (no training)
X <- matrix(rnorm(n * K), n, K)
W0 <- matrix(rnorm(K * 8), K, 8) * 0.3
W1 <- matrix(rnorm(8 * K), 8, K) * 0.3
H1 <- pmax(A_hat %*% X %*% W0, 0)
logits <- A_hat %*% H1 %*% W1
pred <- max.col(logits) - 1L
cat("=== GCN forward pass (Kipf-Welling 2017) ===\n")
cat(sprintf("  n = %d, classes = %d\n", n, K))
cat(sprintf("  Random-init accuracy = %.3f (chance = %.3f)\n",
            mean(pred == y), 1 / K))
cat("For training, use `pytorch-geometric` / `dgl` in Python via reticulate.\n")
