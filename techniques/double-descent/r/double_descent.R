# Double Descent (Reference Sec 47.162).
#
# Belkin et al 2019 / Nakkiran et al 2020. Test error follows a
# double-descent curve as a function of model / data complexity.
#
# Pure-R demo: min-norm least squares sweep.

set.seed(0)
n_tr <- 40; n_te <- 400; d_true <- 10
W_true <- rnorm(d_true)
X_tr_full <- matrix(rnorm(n_tr * 500), n_tr, 500)
X_te_full <- matrix(rnorm(n_te * 500), n_te, 500)
y_tr <- X_tr_full[, 1:d_true] %*% W_true + rnorm(n_tr, sd = 0.5)
y_te <- X_te_full[, 1:d_true] %*% W_true + rnorm(n_te, sd = 0.5)

min_norm_lstsq <- function(X, y, ridge = 1e-8) {
    n <- nrow(X); p <- ncol(X)
    if (p <= n) {
        solve(crossprod(X) + ridge * diag(p), crossprod(X, y))
    } else {
        alpha <- solve(tcrossprod(X) + ridge * diag(n), y)
        t(X) %*% alpha
    }
}

cat("=== Double descent (Belkin et al 2019; Nakkiran et al 2020) ===\n")
cat(sprintf("%4s %10s %10s\n", "p", "train_MSE", "test_MSE"))
for (p in c(2, 5, 10, 20, 30, 39, 40, 41, 50, 100, 300, 500)) {
    W <- min_norm_lstsq(X_tr_full[, 1:p], y_tr)
    tr_mse <- mean((X_tr_full[, 1:p] %*% W - y_tr)^2)
    te_mse <- mean((X_te_full[, 1:p] %*% W - y_te)^2)
    cat(sprintf("%4d %10.4f %10.4f\n", p, tr_mse, te_mse))
}
