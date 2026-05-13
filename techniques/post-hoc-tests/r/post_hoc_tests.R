# Post-hoc pairwise comparisons after a one-way ANOVA (Reference §3.10, §3.11, §3.16)
# From-scratch base R plus idiomatic calls (TukeyHSD, multcomp::glht, PMCMRplus).
# Run with:  Rscript post_hoc_tests.R
#
# Inputs used below:
#   groups        : list of numeric vectors, one per group
#   labels        : optional group labels (default 1..k)
#   control_index : 1-based index of the control group (for Dunnett)
#   conf          : confidence level (e.g. 0.95)

stats_of <- function(g) c(n = length(g), m = mean(g), s2 = var(g))

tukey_hsd_scratch <- function(groups, labels = NULL, conf = 0.95) {
  if (is.null(labels)) labels <- seq_along(groups)
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]
  N <- sum(ns); k <- length(ns); df <- N - k
  ms_w <- sum((ns - 1) * vars) / df
  pairs <- combn(seq_along(groups), 2)
  alpha <- 1 - conf; q_crit <- qtukey(1 - alpha, k, df)
  out <- vector("list", ncol(pairs))
  for (p in seq_len(ncol(pairs))) {
    i <- pairs[1, p]; j <- pairs[2, p]
    diff <- means[i] - means[j]
    se <- sqrt(ms_w * 0.5 * (1/ns[i] + 1/ns[j]))
    q <- abs(diff) / se
    margin <- q_crit * se
    out[[p]] <- list(pair = c(labels[i], labels[j]), mean_diff = diff,
                     se = se, q = q,
                     p_adj = ptukey(q, k, df, lower.tail = FALSE),
                     ci_lower = diff - margin, ci_upper = diff + margin)
  }
  out
}

# Dunnett (vs single control): exact via multcomp; from-scratch falls back to
# Bonferroni-adjusted two-sample t (simpler but less powerful)
dunnett_scratch <- function(groups, control_index = 1, labels = NULL) {
  if (is.null(labels)) labels <- seq_along(groups)
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]
  N <- sum(ns); k <- length(ns); df <- N - k
  ms_w <- sum((ns - 1) * vars) / df
  treats <- setdiff(seq_along(groups), control_index); m <- length(treats)
  out <- vector("list", m)
  for (idx in seq_along(treats)) {
    i <- treats[idx]; j <- control_index
    diff <- means[i] - means[j]
    se <- sqrt(ms_w * (1/ns[i] + 1/ns[j]))
    t <- diff / se
    p <- 2 * pt(-abs(t), df)
    out[[idx]] <- list(pair = c(labels[i], labels[j]),
                       mean_diff = diff, se = se, t = t,
                       p_adj = min(1, p * m),
                       method = "Bonferroni fallback")
  }
  out
}

games_howell_scratch <- function(groups, labels = NULL, conf = 0.95) {
  if (is.null(labels)) labels <- seq_along(groups)
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]
  k <- length(ns); alpha <- 1 - conf
  pairs <- combn(seq_along(groups), 2)
  out <- vector("list", ncol(pairs))
  for (p in seq_len(ncol(pairs))) {
    i <- pairs[1, p]; j <- pairs[2, p]
    diff <- means[i] - means[j]
    se <- sqrt((vars[i]/ns[i] + vars[j]/ns[j]) / 2)
    df <- (vars[i]/ns[i] + vars[j]/ns[j])^2 /
          ((vars[i]/ns[i])^2 / (ns[i] - 1) + (vars[j]/ns[j])^2 / (ns[j] - 1))
    q <- abs(diff) / se
    q_crit <- qtukey(1 - alpha, k, df); margin <- q_crit * se
    out[[p]] <- list(pair = c(labels[i], labels[j]),
                     mean_diff = diff, se = se, df = df, q = q,
                     p_adj = ptukey(q, k, df, lower.tail = FALSE),
                     ci_lower = diff - margin, ci_upper = diff + margin)
  }
  out
}

# Library: stats::TukeyHSD (after aov), multcomp::glht (Dunnett),
#          PMCMRplus::gamesHowellTest
library_demo <- function(groups, labels = NULL) {
  if (is.null(labels)) labels <- seq_along(groups)
  values <- unlist(groups)
  grp <- factor(rep(labels, sapply(groups, length)), levels = labels)
  fit <- aov(values ~ grp)
  cat("TukeyHSD (stats):\n"); print(TukeyHSD(fit))
  if (requireNamespace("multcomp", quietly = TRUE)) {
    cat("\nmultcomp Dunnett (vs first level):\n")
    print(summary(multcomp::glht(fit, linfct = multcomp::mcp(grp = "Dunnett"))))
  }
}

if (sys.nframe() == 0) {
  set.seed(3)
  a <- rnorm(30, 50, 10); b <- rnorm(28, 55, 10)
  c <- rnorm(32, 60, 18); d <- rnorm(30, 52, 10)
  groups <- list(a, b, c, d); labels <- c("A", "B", "C", "D")
  cat("=== Tukey HSD ===\n");    print(tukey_hsd_scratch(groups, labels))
  cat("\n=== Dunnett (vs A) ===\n"); print(dunnett_scratch(groups, control_index = 1, labels))
  cat("\n=== Games-Howell ===\n");  print(games_howell_scratch(groups, labels))
  cat("\n--- library ---\n");      library_demo(groups, labels)
}
