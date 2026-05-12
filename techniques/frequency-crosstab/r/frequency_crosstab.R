# Frequency distributions and cross-tabulations (Reference §1.7)
# From-scratch base-R plus idiomatic base/janitor/gmodels calls.
# Run with:  Rscript frequency_crosstab.R

frequency_table <- function(x) {
  tab <- table(x); n <- sum(tab)
  data.frame(category = names(tab), count = as.integer(tab),
             rel_freq = as.numeric(tab) / n,
             cum_count = cumsum(as.integer(tab)),
             cum_rel = cumsum(as.numeric(tab)) / n, row.names = NULL)
}

crosstab_scratch <- function(row, col) {
  rl <- sort(unique(row)); cl <- sort(unique(col))
  m <- matrix(0L, length(rl), length(cl), dimnames = list(rl, cl))
  for (i in seq_along(row)) m[as.character(row[i]), as.character(col[i])] <-
    m[as.character(row[i]), as.character(col[i])] + 1L
  m
}

with_margins <- function(m) addmargins(m)

percentages <- function(m, kind = c("total", "row", "col")) {
  kind <- match.arg(kind)
  switch(kind,
    total = m / sum(m) * 100,
    row   = sweep(m, 1, rowSums(m), "/") * 100,
    col   = sweep(m, 2, colSums(m), "/") * 100)
}

# Numeric binning rules (same as a histogram): Sturges / Scott / Freedman-Diaconis
bin_counts <- function(x, rule = c("FD", "Sturges", "Scott")) {
  rule <- match.arg(rule)
  br <- switch(rule, FD = "FD", Sturges = "Sturges", Scott = "Scott")
  h <- hist(x, breaks = br, plot = FALSE)
  data.frame(lo = head(h$breaks, -1), hi = tail(h$breaks, -1), count = h$counts)
}

# Library: table(), prop.table(), xtabs(), addmargins(), janitor::tabyl,
#          gmodels::CrossTable (SPSS/SAS-style with row/col/total %)
library_demo <- function(row, col) {
  cat("table():\n"); print(table(row))
  cat("\nprop.table (relative freq):\n"); print(round(prop.table(table(row)), 3))
  cat("\nxtabs + addmargins:\n"); print(addmargins(xtabs(~ row + col)))
  cat("\nrow proportions:\n"); print(round(prop.table(xtabs(~ row + col), margin = 1), 3))
}

if (sys.nframe() == 0) {
  set.seed(7)
  region  <- sample(c("North", "South", "East"), 60, replace = TRUE, prob = c(.5, .3, .2))
  outcome <- sample(c("Yes", "No"), 60, replace = TRUE, prob = c(.4, .6))
  cat("=== frequency table: region ===\n"); print(frequency_table(region))
  m <- crosstab_scratch(region, outcome)
  cat("\n=== cross-tab with margins ===\n"); print(with_margins(m))
  cat("\n=== row percentages ===\n"); print(round(percentages(m, "row"), 1))
  cat("\n=== binned numeric (FD) ===\n"); print(bin_counts(rnorm(200, 50, 10), "FD"))
  cat("\n--- library ---\n"); library_demo(region, outcome)
}
