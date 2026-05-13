# Variable selection in regression (Reference §5.8, §5.18, §5.19, §5.36)
# From-scratch base R plus stats::step / leaps::regsubsets.
# Run with:  Rscript variable_selection.R
#
# Inputs used below:
#   X    : n x p design matrix (no intercept column; we add one)
#   y    : response vector
#   kind : "AIC" / "BIC" / "adj_R2"  (criterion to minimize)

fit_rss <- function(X, y) {
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  sum((y - as.vector(X %*% beta))^2)
}

criterion_scratch <- function(X, y, kind = "AIC") {
  n <- nrow(X); p <- ncol(X); rss <- fit_rss(X, y)
  switch(kind,
    "AIC"    = n * log(rss / n) + 2 * p,
    "BIC"    = n * log(rss / n) + log(n) * p,
    "adj_R2" = -(1 - (1 - (1 - rss / sum((y - mean(y))^2))) * (n - 1) / (n - p)))
}

.design <- function(X_full, idx) cbind(1, X_full[, idx, drop = FALSE])

forward_stepwise_scratch <- function(X, y, kind = "AIC", verbose = FALSE) {
  X <- as.matrix(X); p <- ncol(X)
  selected <- integer(0); remaining <- seq_len(p)
  best_crit <- criterion_scratch(.design(X, selected), y, kind)
  history <- list(list(step = 0, selected = selected, criterion = best_crit))
  while (length(remaining) > 0) {
    cand_scores <- sapply(remaining, function(j)
      criterion_scratch(.design(X, c(selected, j)), y, kind))
    j_best <- remaining[which.min(cand_scores)]
    c_new <- min(cand_scores)
    if (c_new < best_crit) {
      best_crit <- c_new; selected <- c(selected, j_best)
      remaining <- setdiff(remaining, j_best)
      history[[length(history) + 1]] <- list(
        step = length(selected), selected = selected,
        criterion = c_new, added = j_best)
      if (verbose) cat(sprintf("  + j=%d  -> %s = %.4f\n", j_best, kind, c_new))
    } else break
  }
  list(selected = selected, best_criterion = best_crit, history = history,
       criterion_kind = kind)
}

backward_stepwise_scratch <- function(X, y, kind = "AIC", verbose = FALSE) {
  X <- as.matrix(X); p <- ncol(X)
  selected <- seq_len(p)
  best_crit <- criterion_scratch(.design(X, selected), y, kind)
  while (length(selected) > 0) {
    cand_scores <- sapply(selected, function(j)
      criterion_scratch(.design(X, setdiff(selected, j)), y, kind))
    j_drop <- selected[which.min(cand_scores)]
    c_new <- min(cand_scores)
    if (c_new < best_crit) {
      best_crit <- c_new; selected <- setdiff(selected, j_drop)
      if (verbose) cat(sprintf("  - j=%d  -> %s = %.4f\n", j_drop, kind, c_new))
    } else break
  }
  list(selected = selected, best_criterion = best_crit, criterion_kind = kind)
}

best_subsets_scratch <- function(X, y, kind = "AIC", max_size = NULL) {
  X <- as.matrix(X); p <- ncol(X)
  if (is.null(max_size)) max_size <- p
  best_per_size <- list(); overall <- list(crit = Inf, subset = NULL)
  for (k in 0:max_size) {
    best_here <- list(crit = Inf, subset = NULL)
    if (k == 0) {
      c <- criterion_scratch(.design(X, integer(0)), y, kind)
      best_here <- list(crit = c, subset = integer(0))
    } else {
      subs <- combn(p, k); for (s in seq_len(ncol(subs))) {
        idx <- subs[, s]
        c <- criterion_scratch(.design(X, idx), y, kind)
        if (c < best_here$crit) best_here <- list(crit = c, subset = idx)
      }
    }
    best_per_size[[k + 1]] <- best_here
    if (best_here$crit < overall$crit) overall <- best_here
  }
  list(best_per_size = best_per_size, overall = overall, criterion_kind = kind)
}

# Library: stats::step (forward/backward via AIC), leaps::regsubsets (best subsets)
library_demo <- function(df) {
  fit_full <- lm(y ~ ., data = df); fit_null <- lm(y ~ 1, data = df)
  cat("step(forward):\n")
  print(step(fit_null, scope = formula(fit_full), direction = "forward", trace = 0))
  cat("\nstep(backward):\n")
  print(step(fit_full, direction = "backward", trace = 0))
  if (requireNamespace("leaps", quietly = TRUE)) {
    cat("\nleaps::regsubsets:\n")
    rs <- leaps::regsubsets(y ~ ., data = df, nvmax = ncol(df) - 1)
    print(summary(rs)$which)
  }
}

if (sys.nframe() == 0) {
  set.seed(5); n <- 100; p <- 8
  X <- matrix(rnorm(n * p), n, p)
  true_beta <- c(2, -1.5, 0, 0, 1, 0, 0, -0.3)
  y <- as.vector(X %*% true_beta) + rnorm(n)
  cat("=== Forward stepwise (AIC) ===\n")
  print(forward_stepwise_scratch(X, y, kind = "AIC", verbose = TRUE)$selected)
  cat("\n=== Backward stepwise (AIC) ===\n")
  print(backward_stepwise_scratch(X, y, kind = "AIC", verbose = TRUE)$selected)
  cat("\n=== Best subsets (BIC, max 5) ===\n")
  print(best_subsets_scratch(X, y, kind = "BIC", max_size = 5)$overall)
  cat(sprintf("\nTrue non-zero indices: %s\n",
              paste(which(true_beta != 0), collapse = ", ")))
  cat("\n--- library ---\n")
  df <- data.frame(y = y, setNames(as.data.frame(X), paste0("x", 1:p)))
  library_demo(df)
}
