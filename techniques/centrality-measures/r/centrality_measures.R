# Centrality measures (Reference §24.2)
# R via igraph.
# Run with:  Rscript centrality_measures.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::degree(g, normalized=TRUE)                   -- degree\n")
  cat("  igraph::closeness(g, normalized=TRUE)                -- closeness\n")
  cat("  igraph::betweenness(g, normalized=TRUE)              -- Brandes betweenness\n")
  cat("  igraph::eigen_centrality(g)                          -- Perron eigenvector\n")
  cat("  igraph::alpha_centrality(g, alpha)                   -- Katz-style\n")
  cat("  igraph::page_rank(g, damping=0.85)                   -- PageRank\n")
  cat("  igraph::authority_score(g) / hub_score(g)            -- HITS\n")
}
