# Patient similarity network (Reference Sec 30.25)
# Native R via SNFtool; Python via networkx + community detection.
# Run with:  Rscript patient_similarity_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SNFtool                      -- Wang et al. Similarity Network Fusion\n")
  cat("  igraph::cluster_louvain       -- Louvain modularity community detection\n")
  cat("  bootnet, qgraph               -- adjacent network psychometrics tools\n")
  cat("Python:\n")
  cat("  snfpy                         -- Similarity Network Fusion in Python\n")
  cat("  python-louvain                -- community detection (Blondel 2008)\n")
  cat("  networkx.algorithms.community -- label propagation, greedy modularity, LPA\n")
  cat("  scikit-network                -- fast Louvain + K-medoids on graphs\n")
  cat("Refs: Wang, B. et al. (2014) 'Similarity network fusion for aggregating data\n")
  cat("      types on a genomic scale', Nature Methods;\n")
  cat("      Li, L. et al. (2015) 'Identification of type 2 diabetes subgroups through\n")
  cat("      topological analysis of patient similarity', Science Translational Medicine.\n")
}
