# Graph descriptive statistics (Reference §24.1)
# R via igraph.
# Run with:  Rscript graph_descriptives.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::edge_density(g)                            -- density\n")
  cat("  igraph::degree(g)  igraph::degree_distribution(g)  -- degree summary\n")
  cat("  igraph::transitivity(g, 'local' | 'global')        -- clustering / transitivity\n")
  cat("  igraph::assortativity_degree(g)                    -- Newman assortativity\n")
  cat("  igraph::mean_distance(g)  igraph::diameter(g)      -- path summaries\n")
  cat("  igraph::components(g)                              -- connected components\n")
}
