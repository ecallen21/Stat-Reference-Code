# Collinearity diagnostics for OLS regression (Reference §5.23)
# From-scratch base R plus car::vif. Run with:  Rscript collinearity_diagnostics.R
#
# Inputs used below:
#   X     : n x p design matrix (first column = 1s for the intercept)
#   names : optional column labels

pairwise_correlations_scratch <- function(X, names = NULL) {
  X <- as.matrix(X); has_int <- isTRUE(all.equal(X[, 1], rep(1, nrow(X))))
  Xc <- if (has_int) X[, -1, drop = FALSE] else X
  if (is.null(names)) names <- paste0("x", seq_len(ncol(Xc)))
  list(names = names, correlation = cor(Xc))
}

vif_scratch <- function(X, names = NULL) {
  X <- as.matrix(X); has_int <- isTRUE(all.equal(X[, 1], rep(1, nrow(X))))
  Xc <- if (has_int) X[, -1, drop = FALSE] else X
  k <- ncol(Xc); if (is.null(names)) names <- paste0("x", seq_len(k))
  v <- numeric(k); tol <- numeric(k)
  for (j in seq_len(k)) {
    y <- Xc[, j]; Xj <- cbind(1, Xc[, -j, drop = FALSE])
    fit <- lm(y ~ Xj - 1)
    r2 <- summary(fit)$r.squared
    v[j] <- 1 / max(1e-15, 1 - r2); tol[j] <- 1 / v[j]
  }
  list(names = names, vif = v, tolerance = tol)
}

condition_indices_scratch <- function(X, names = NULL) {
  X <- as.matrix(X); p <- ncol(X); n <- nrow(X)
  if (is.null(names)) names <- ifelse(apply(X, 2, function(c) isTRUE(all.equal(c, rep(1, n)))),
                                       "intercept",
                                       paste0("x", seq_len(p)))
  norms <- sqrt(colSums(X^2))
  Xs <- sweep(X, 2, ifelse(norms == 0, 1, norms), "/")
  sv <- svd(Xs)
  sing <- pmax(sv$d, 1e-15); kappa <- max(sing) / sing
  raw <- (sv$v^2) / (sing^2)[col(sv$v)]
  proportions <- raw / rowSums(raw)
  list(names = names, singular_values = sing,
       condition_indices = kappa, max_condition_index = max(kappa),
       variance_proportions = proportions)
}

# Library: car::vif (most idiomatic), perturb::colldiag for condition indices
library_demo <- function(df) {
  fit <- lm(y ~ x1 + x2 + x3, data = df)
  if (requireNamespace("car", quietly = TRUE)) {
    cat("car::vif:\n"); print(car::vif(fit))
  } else cat("(install 'car' for vif)\n")
  if (requireNamespace("perturb", quietly = TRUE)) {
    cat("\nperturb::colldiag:\n"); print(perturb::colldiag(fit))
  }
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 80
  x1 <- rnorm(n); x2 <- 0.95 * x1 + rnorm(n, 0, 0.2); x3 <- rnorm(n)
  X <- cbind(1, x1, x2, x3)
  cat("=== Pairwise correlations ===\n")
  print(pairwise_correlations_scratch(X, c("x1", "x2", "x3")))
  cat("\n=== VIF ===\n"); print(vif_scratch(X, c("x1", "x2", "x3")))
  cat("\n=== Condition indices ===\n");
  print(condition_indices_scratch(X, c("intercept", "x1", "x2", "x3")))
  cat("\n--- library ---\n")
  library_demo(data.frame(y = 1 + 2 * x1 - x2 + x3 + rnorm(n),
                          x1 = x1, x2 = x2, x3 = x3))
}
