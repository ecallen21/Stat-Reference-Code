# Standardized regression coefficients and dominance analysis (Reference §5.38)
# From-scratch base R plus lm.beta::lm.beta / relaimpo::calc.relimp.
# Run with:  Rscript standardized_coefficients.R
#
# Inputs used below:
#   X     : n x p matrix of predictors
#   y     : response vector
#   names : optional column labels

ols_r2 <- function(X, y) {
  if (ncol(X) == 0) return(0)
  Xc <- cbind(1, X)
  beta <- as.vector(solve(crossprod(Xc), crossprod(Xc, y)))
  rss <- sum((y - as.vector(Xc %*% beta))^2)
  tss <- sum((y - mean(y))^2)
  1 - rss / tss
}

standardized_coefficients_scratch <- function(X, y) {
  mu_x <- colMeans(X); sd_x <- apply(X, 2, sd)
  mu_y <- mean(y); sd_y <- sd(y)
  Xz <- sweep(sweep(X, 2, mu_x), 2, sd_x, "/")
  yz <- (y - mu_y) / sd_y
  Xc <- cbind(1, Xz)
  beta_std <- as.vector(solve(crossprod(Xc), crossprod(Xc, yz)))
  Xc_raw <- cbind(1, X)
  beta_raw <- as.vector(solve(crossprod(Xc_raw), crossprod(Xc_raw, y)))
  list(intercept_standardized = beta_std[1],
       beta_standardized = beta_std[-1],
       beta_raw = beta_raw[-1],
       sd_x = sd_x, sd_y = sd_y)
}

dominance_analysis_scratch <- function(X, y, names = NULL) {
  p <- ncol(X)
  if (is.null(names)) names <- paste0("x", seq_len(p))
  total_r2 <- ols_r2(X, y)
  contributions <- vector("list", p); for (j in seq_len(p)) contributions[[j]] <- list()
  for (k in 0:(p - 1)) {
    subs <- if (k == 0) list(integer(0)) else combn(p, k, simplify = FALSE)
    for (S in subs) {
      r2_S <- ols_r2(X[, S, drop = FALSE], y)
      for (j in setdiff(seq_len(p), S)) {
        r2_Sj <- ols_r2(X[, c(S, j), drop = FALSE], y)
        contributions[[j]][[length(contributions[[j]]) + 1]] <-
          list(size = k, gain = r2_Sj - r2_S)
      }
    }
  }
  gd <- vapply(seq_len(p), function(j) {
    by_size <- split(sapply(contributions[[j]], `[[`, "gain"),
                     sapply(contributions[[j]], `[[`, "size"))
    mean(sapply(by_size, mean))
  }, numeric(1))
  list(names = names,
       general_dominance = setNames(gd, names),
       total_r_squared = total_r2,
       sum_of_dominance = sum(gd))
}

# Library: QuantPsyc::lm.beta (or lm.beta::lm.beta) for standardized betas,
#          relaimpo::calc.relimp(..., type = "lmg") for dominance / LMG decomposition
library_demo <- function(X, y) {
  fit <- lm(y ~ X)
  if (requireNamespace("lm.beta", quietly = TRUE))
    print(lm.beta::lm.beta(fit))
  else cat("(install 'lm.beta' for standardized betas)\n")
  if (requireNamespace("relaimpo", quietly = TRUE)) {
    cat("\nrelaimpo::calc.relimp (lmg = general-dominance equivalent):\n")
    print(relaimpo::calc.relimp(fit, type = "lmg"))
  }
}

if (sys.nframe() == 0) {
  set.seed(12); n <- 200; p <- 4
  A <- matrix(c(1, .7, 0, 0, .7, 1, 0, 0, 0, 0, 1, .3, 0, 0, .3, 1), nrow = 4)
  L <- chol(A); X <- matrix(rnorm(n * p), n, p) %*% L
  true_beta <- c(1, 0.5, 0.8, -0.4)
  y <- as.vector(X %*% true_beta) + rnorm(n)
  cat("=== Standardized coefficients ===\n")
  print(standardized_coefficients_scratch(X, y))
  cat("\n=== Dominance analysis ===\n")
  print(dominance_analysis_scratch(X, y))
  cat("\n--- library ---\n"); library_demo(X, y)
}
