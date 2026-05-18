# Splines and segmented regression (Reference §5.4, §5.22, §5.26, §5.34)
# From-scratch base R plus splines::ns / segmented::segmented.
# Run with:  Rscript splines_segmented.R
#
# Inputs used below:
#   x, y  : numeric vectors
#   knots : knot locations (in x-units)

piecewise_linear_basis_scratch <- function(x, knots) {
  cbind(1, x, sapply(knots, function(k) pmax(0, x - k)))
}

fit_piecewise_linear_scratch <- function(x, y, knots) {
  X <- piecewise_linear_basis_scratch(x, knots)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  yhat <- as.vector(X %*% beta)
  list(beta = beta, knots = knots, fitted = yhat,
       rss = sum((y - yhat)^2))
}

search_breakpoint_scratch <- function(x, y, n_grid = 50) {
  candidates <- seq(quantile(x, 0.1), quantile(x, 0.9), length.out = n_grid)
  best <- list(rss = Inf, breakpoint = NA, fit = NULL)
  for (k in candidates) {
    fit <- fit_piecewise_linear_scratch(x, y, knots = k)
    if (fit$rss < best$rss) best <- list(rss = fit$rss, breakpoint = k, fit = fit)
  }
  best
}

natural_cubic_basis_scratch <- function(x, knots) {
  knots <- sort(knots); K <- length(knots); stopifnot(K >= 3)
  d <- function(j) (pmax(0, x - knots[j])^3 - pmax(0, x - knots[K])^3) /
                    (knots[K] - knots[j])
  cols <- cbind(1, x)
  dK1 <- d(K - 1)
  for (j in seq_len(K - 2)) cols <- cbind(cols, d(j) - dK1)
  cols
}

fit_natural_cubic_spline_scratch <- function(x, y, knots) {
  X <- natural_cubic_basis_scratch(x, knots); n <- nrow(X); p <- ncol(X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  yhat <- as.vector(X %*% beta); rss <- sum((y - yhat)^2)
  tss <- sum((y - mean(y))^2)
  list(beta = beta, knots = knots, fitted = yhat, rss = rss, df = p,
       r_squared = 1 - rss / tss)
}

# Library: splines::ns (natural cubic spline basis), splines::bs (B-spline),
#          rms::rcs (restricted cubic spline), segmented::segmented (breakpoint)
library_demo <- function(x, y) {
  if (requireNamespace("splines", quietly = TRUE)) {
    cat("lm with splines::ns (df = 4):\n")
    fit <- lm(y ~ splines::ns(x, df = 4))
    print(summary(fit))
  }
  if (requireNamespace("segmented", quietly = TRUE)) {
    cat("\nsegmented::segmented:\n")
    lin <- lm(y ~ x)
    print(segmented::segmented(lin, seg.Z = ~x, psi = 5))
  }
}

if (sys.nframe() == 0) {
  set.seed(8); n <- 200
  x <- seq(0, 10, length.out = n)
  y_true <- ifelse(x < 4, 0.5 + 1.0 * x, 4.5 - 0.5 * (x - 4))
  y <- y_true + rnorm(n, 0, 0.3)
  cat("=== Piecewise linear, known knot at x = 4 ===\n")
  print(fit_piecewise_linear_scratch(x, y, knots = 4))
  cat("\n=== Breakpoint search ===\n")
  bk <- search_breakpoint_scratch(x, y); cat("  breakpoint =", bk$breakpoint, "  RSS =", bk$rss, "\n")
  cat("\n=== Natural cubic spline (5 knots at quantiles) ===\n")
  knots <- quantile(x, c(.1, .3, .5, .7, .9))
  print(fit_natural_cubic_spline_scratch(x, y, knots = knots)[c("beta", "r_squared", "df")])
  cat("\n--- library ---\n"); library_demo(x, y)
}
