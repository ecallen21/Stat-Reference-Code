# Community detection (Reference §24.3)
# R via igraph.
# Run with:  Rscript community_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::cluster_louvain(g)                          -- Louvain modularity\n")
  cat("  igraph::cluster_leiden(g, objective='modularity')   -- Leiden (recommended)\n")
  cat("  igraph::cluster_fast_greedy(g)                      -- Clauset-Newman-Moore\n")
  cat("  igraph::cluster_walktrap(g, steps=4)                -- random-walk based\n")
  cat("  igraph::cluster_infomap(g)                          -- MDL-based, Rosvall-Bergstrom\n")
  cat("  igraph::modularity(g, membership)                   -- score a partition\n")
  cat("  igraph::compare(m1, m2, method='nmi')               -- NMI vs truth\n")
}
