# Categorical variable coding schemes (Reference §5.32)
# From-scratch base R plus stats::contr.treatment / contr.sum / contr.helmert.
# Run with:  Rscript categorical_variable_coding.R
#
# Inputs used below:
#   group : character / factor vector giving the level of each observation
#   y     : numeric response

dummy_coding <- function(group, reference = NULL) {
  levels <- sort(unique(group))
  if (is.null(reference)) reference <- levels[1]
  others <- setdiff(levels, reference)
  X <- sapply(others, function(lev) as.numeric(group == lev))
  list(X = X, names = paste0(others, "_vs_", reference),
       meta = list(reference = reference, scheme = "dummy"))
}

effect_coding <- function(group, reference = NULL) {
  levels <- sort(unique(group))
  if (is.null(reference)) reference <- levels[length(levels)]
  others <- setdiff(levels, reference)
  X <- sapply(others, function(lev) ifelse(group == lev, 1,
                                           ifelse(group == reference, -1, 0)))
  list(X = X, names = paste0(others, "_vs_grandmean"),
       meta = list(reference = reference, scheme = "effect (sum-to-zero)"))
}

helmert_coding <- function(group, levels = NULL) {
  if (is.null(levels)) levels <- sort(unique(group))
  k <- length(levels); cols <- list(); names <- character(0)
  for (j in 2:k) {
    col <- numeric(length(group))
    col[group == levels[j]] <- j - 1
    col[group %in% levels[seq_len(j - 1)]] <- -1
    col <- col / j                                  # scale for orthogonality
    cols[[length(cols) + 1]] <- col
    names <- c(names, paste0(levels[j], "_vs_avg_of_prev"))
  }
  X <- do.call(cbind, cols)
  list(X = X, names = names, meta = list(levels_order = levels, scheme = "Helmert"))
}

deviation_coding <- function(group, omitted = NULL) {
  levels <- sort(unique(group))
  if (is.null(omitted)) omitted <- levels[length(levels)]
  others <- setdiff(levels, omitted)
  X <- sapply(others, function(lev) ifelse(group == lev, 1,
                                           ifelse(group == omitted, -1, 0)))
  list(X = X, names = paste0(others, "_vs_grandmean"),
       meta = list(omitted = omitted, scheme = "deviation"))
}

fit_one_factor <- function(group, y, coding_fn) {
  cd <- coding_fn(group); X <- cbind(1, cd$X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  yhat <- as.vector(X %*% beta); rss <- sum((y - yhat)^2)
  tss <- sum((y - mean(y))^2)
  list(scheme = cd$meta$scheme, intercept = beta[1],
       coefficients = setNames(beta[-1], cd$names),
       r_squared = 1 - rss / tss)
}

# Library: in R, the default 'contrasts' option controls coding; set via
# options(contrasts = c("contr.treatment", "contr.poly"))  -- the first is for unordered.
# Inside formulas use C(factor, contr.treatment / contr.sum / contr.helmert)
library_demo <- function(group, y) {
  g <- factor(group)
  cat("contr.treatment (= our 'dummy'):\n")
  print(coef(lm(y ~ C(g, contr.treatment))))
  cat("\ncontr.sum (= our 'effect'):\n")
  print(coef(lm(y ~ C(g, contr.sum))))
  cat("\ncontr.helmert (= our 'Helmert', possibly scaled differently):\n")
  print(coef(lm(y ~ C(g, contr.helmert))))
}

if (sys.nframe() == 0) {
  set.seed(11); n_per <- 30
  means <- c(A = 50, B = 55, C = 60, D = 52)
  group <- rep(names(means), each = n_per)
  y <- as.numeric(unlist(lapply(means, function(m) rnorm(n_per, m, 3))))
  for (fn in list(dummy_coding, effect_coding, helmert_coding, deviation_coding)) {
    res <- fit_one_factor(group, y, fn)
    cat("===", res$scheme, "===\n")
    cat(sprintf("  intercept = %.4f\n", res$intercept))
    print(round(res$coefficients, 4))
    cat(sprintf("  R^2 = %.4f\n\n", res$r_squared))
  }
  cat("--- library ---\n"); library_demo(group, y)
}
