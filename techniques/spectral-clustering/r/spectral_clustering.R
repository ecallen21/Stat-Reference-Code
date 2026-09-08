# Spectral clustering (Reference Sec 47.78)
# Native R via kernlab / RSpectra; Python via sklearn.
# Run with:  Rscript spectral_clustering.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kernlab::specc                -- Ng-Jordan-Weiss + RBF affinity\n")
  cat("  RSpectra::eigs                -- ARPACK top / bottom eigenvectors\n")
  cat("  igraph::cluster_leading_eigen -- spectral community detection on graphs\n")
  cat("Python:\n")
  cat("  sklearn.cluster.SpectralClustering    -- kNN / RBF affinity, precomputed\n")
  cat("  scipy.sparse.linalg.eigsh             -- sparse spectral decomposition\n")
  cat("  from-scratch                          -- see spectral_clustering.py\n")
  cat("Refs: Ng, Jordan & Weiss (2001) NeurIPS; Shi & Malik (2000) IEEE TPAMI\n")
  cat("      22(8); Von Luxburg (2007) Stat Comput 17(4).\n")
}
