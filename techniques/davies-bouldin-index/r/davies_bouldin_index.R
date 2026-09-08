# Davies-Bouldin index (Reference Sec 47.82)
# Native R via fpc / clusterCrit; Python via sklearn.
# Run with:  Rscript davies_bouldin_index.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fpc::cluster.stats               -- DB + Dunn + CH + more\n")
  cat("  clusterCrit::intCriteria         -- DB + 42 other internal criteria\n")
  cat("  NbClust::NbClust                 -- pooled cluster-index voting\n")
  cat("Python:\n")
  cat("  sklearn.metrics.davies_bouldin_score\n")
  cat("  yellowbrick.cluster.KElbow(metric='davies_bouldin')\n")
  cat("  from-scratch                      -- see davies_bouldin_index.py\n")
  cat("Refs: Davies & Bouldin (1979) 'A cluster separation measure', IEEE TPAMI\n")
  cat("      PAMI-1(2).\n")
}
