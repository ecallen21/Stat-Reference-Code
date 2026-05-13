# Multiple-comparison p-value adjustments (Reference §3.13, §3.14)
# From-scratch base R plus stats::p.adjust.  Run with:  Rscript multiple_comparisons.R
#
# Inputs used below:
#   p      : numeric vector of raw p-values from m hypothesis tests
#   method : "bonferroni" / "sidak" / "holm" / "hochberg" / "bh" / "fdr" / "by"

bonferroni_scratch <- function(p) pmin(1, length(p) * p)
sidak_scratch      <- function(p) pmin(1, 1 - (1 - p)^length(p))

holm_scratch <- function(p) {                       # step-down
  m <- length(p); ord <- order(p); adj <- numeric(m); running_max <- 0
  for (rank in seq_len(m)) {
    cand <- (m - rank + 1) * p[ord[rank]]
    running_max <- max(running_max, cand); adj[ord[rank]] <- min(1, running_max)
  }
  adj
}

hochberg_scratch <- function(p) {                   # step-up
  m <- length(p); ord <- order(p, decreasing = TRUE)
  adj <- numeric(m); running_min <- 1
  for (rank_from_top in seq_len(m)) {
    i <- m - rank_from_top + 1
    cand <- (m - i + 1) * p[ord[rank_from_top]]
    running_min <- min(running_min, cand)
    adj[ord[rank_from_top]] <- min(1, running_min)
  }
  adj
}

bh_scratch <- function(p) {                         # Benjamini-Hochberg FDR
  m <- length(p); ord <- order(p, decreasing = TRUE)
  adj <- numeric(m); running_min <- 1
  for (rank_from_top in seq_len(m)) {
    i <- m - rank_from_top + 1
    cand <- (m / i) * p[ord[rank_from_top]]
    running_min <- min(running_min, cand)
    adj[ord[rank_from_top]] <- min(1, running_min)
  }
  adj
}

by_scratch <- function(p) {                         # Benjamini-Yekutieli FDR
  m <- length(p); cm <- sum(1 / seq_len(m))
  pmin(1, bh_scratch(p) * cm)
}

adjust <- function(p, method = "holm") {
  method <- tolower(method)
  switch(method,
    bonferroni = bonferroni_scratch(p),
    sidak      = sidak_scratch(p),
    holm       = holm_scratch(p),
    hochberg   = hochberg_scratch(p),
    bh = ,
    fdr        = bh_scratch(p),
    by         = by_scratch(p),
    stop("unknown method: ", method))
}

# Library: stats::p.adjust (note: 'BH' / 'fdr', 'BY', 'holm', 'hochberg',
# 'hommel', 'bonferroni' are the supported method strings)
library_demo <- function(p) {
  for (m in c("bonferroni", "holm", "hochberg", "BH", "BY")) {
    cat(sprintf("%-11s: %s\n", m,
                paste(round(p.adjust(p, method = m), 4), collapse = " ")))
  }
}

if (sys.nframe() == 0) {
  raw_p <- c(0.001, 0.008, 0.039, 0.041, 0.042, 0.30, 0.45, 0.78)
  for (m in c("bonferroni", "sidak", "holm", "hochberg", "bh", "by")) {
    adj <- adjust(raw_p, m)
    cat(sprintf("%-11s: %s   reject@0.05: %s\n",
                m, paste(round(adj, 4), collapse = " "),
                paste(which(adj <= 0.05), collapse = " ")))
  }
  cat("\n--- library (stats::p.adjust) ---\n"); library_demo(raw_p)
}
