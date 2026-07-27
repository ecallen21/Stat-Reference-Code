# Linear + Quadratic Discriminant Analysis (Reference §9.30)
# From-scratch base R + MASS::lda / MASS::qda as library cross-checks.
# Run with:  Rscript lda_qda.R

fit_lda <- function(X, y) {
  X <- as.matrix(X); classes <- sort(unique(y)); K <- length(classes); n <- nrow(X); p <- ncol(X)
  counts <- table(factor(y, levels = classes))
  priors <- counts / n
  means <- t(sapply(classes, function(c) colMeans(X[y == c, , drop = FALSE])))
  Sigma <- matrix(0, p, p)
  for (i in seq_along(classes)) {
    Xc <- X[y == classes[i], , drop = FALSE]
    d <- sweep(Xc, 2, means[i, ]); Sigma <- Sigma + t(d) %*% d
  }
  Sigma <- Sigma / (n - K)
  list(classes = classes, priors = as.numeric(priors), means = means, Sigma = Sigma)
}

predict_lda <- function(fit, Xnew) {
  Xnew <- as.matrix(Xnew); Si <- solve(fit$Sigma)
  d <- sapply(seq_along(fit$classes), function(k) {
    m <- fit$means[k, ]
    Xnew %*% Si %*% m - 0.5 * as.numeric(t(m) %*% Si %*% m) + log(fit$priors[k])
  })
  fit$classes[apply(d, 1, which.max)]
}

fit_qda <- function(X, y) {
  X <- as.matrix(X); classes <- sort(unique(y))
  means <- t(sapply(classes, function(c) colMeans(X[y == c, , drop = FALSE])))
  Sigmas <- lapply(classes, function(c) cov(X[y == c, , drop = FALSE]))
  priors <- as.numeric(table(factor(y, levels = classes)) / nrow(X))
  list(classes = classes, priors = priors, means = means, Sigmas = Sigmas)
}

predict_qda <- function(fit, Xnew) {
  Xnew <- as.matrix(Xnew)
  d <- sapply(seq_along(fit$classes), function(k) {
    S <- fit$Sigmas[[k]]; Si <- solve(S); m <- fit$means[k, ]
    diff <- sweep(Xnew, 2, m)
    -0.5 * as.numeric(determinant(S, logarithm = TRUE)$modulus) -
      0.5 * rowSums((diff %*% Si) * diff) + log(fit$priors[k])
  })
  fit$classes[apply(d, 1, which.max)]
}

if (sys.nframe() == 0) {
  set.seed(89); n_per <- 80
  mus <- rbind(c(0, 0), c(3, 3), c(6, 0))
  Sigma <- matrix(c(1, 0.3, 0.3, 1), 2)
  X <- do.call(rbind, lapply(seq_len(nrow(mus)), function(i) MASS::mvrnorm(n_per, mus[i, ], Sigma)))
  y <- rep(0:2, each = n_per)
  lda_fit <- fit_lda(X, y); pl <- predict_lda(lda_fit, X)
  qda_fit <- fit_qda(X, y); pq <- predict_qda(qda_fit, X)
  cat("LDA acc:", mean(pl == y), "\n")
  cat("QDA acc:", mean(pq == y), "\n")

  if (requireNamespace("MASS", quietly = TRUE)) {
    cat("\n--- library: MASS::lda / MASS::qda ---\n")
    lda_l <- MASS::lda(y ~ X); pl_l <- predict(lda_l)$class
    qda_l <- MASS::qda(y ~ X); pq_l <- predict(qda_l)$class
    cat("MASS LDA acc:", mean(pl_l == y), "\n")
    cat("MASS QDA acc:", mean(pq_l == y), "\n")
  }
}
