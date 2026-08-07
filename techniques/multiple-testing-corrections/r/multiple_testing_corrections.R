# Multiple-testing corrections (Reference §3.30, §4.24)
# Base R p.adjust; Storey q via qvalue package.
# Run with:  Rscript multiple_testing_corrections.R

if (sys.nframe() == 0) {
  set.seed(0)
  p <- c(runif(40), rbeta(10, 0.3, 20))
  cat("=== Base R p.adjust ===\n")
  for (m in c("bonferroni", "holm", "hochberg", "BH", "BY")) {
    adj <- p.adjust(p, method = m)
    cat(sprintf("  %-10s rejects = %d\n", m, sum(adj < 0.05)))
  }
  if (requireNamespace("qvalue", quietly = TRUE)) {
    q <- qvalue::qvalue(p)
    cat(sprintf("\n  Storey q  rejects = %d  (pi0 hat = %.3f)\n",
                sum(q$qvalues < 0.05), q$pi0))
  }
}
