# Non-negative matrix factorisation (Reference Sec 25.2)
# Native R via NMF package; Python via sklearn.
# Run with:  Rscript nmf.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  NMF                          -- Gaujoux-Seoighe reference: multiple algorithms\n")
  cat("  RcppML                       -- fast large-scale NMF\n")
  cat("  MetaGxOvarian, MOFA           -- multi-omics factor analysis\n")
  cat("Python:\n")
  cat("  sklearn.decomposition.NMF     -- multiplicative + coordinate-descent\n")
  cat("  nimfa                          -- extensive NMF algorithm collection\n")
  cat("  tensorly                       -- tensor NMF + CP/Tucker\n")
  cat("Refs: Lee, D.D. & Seung, H.S. (1999) 'Learning the parts of objects by\n")
  cat("      non-negative matrix factorization', Nature 401.\n")
}
