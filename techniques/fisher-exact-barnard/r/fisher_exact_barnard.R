# Fisher's exact and Barnard's exact tests for 2x2 tables (Reference §8.4)
# Base R fisher.test + Exact::exact.test for Barnard.
# Run with:  Rscript fisher_exact_barnard.R

if (sys.nframe() == 0) {
  tbl <- matrix(c(8, 2, 3, 10), nrow = 2, byrow = TRUE)
  cat("=== Fisher's exact ===\n")
  print(fisher.test(tbl))

  if (requireNamespace("Exact", quietly = TRUE)) {
    cat("\n=== Barnard's exact (Exact::exact.test) ===\n")
    print(Exact::exact.test(tbl, method = "Z-pooled"))
  }
}
