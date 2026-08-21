# Random-graph models (Reference §24.4)
# R via igraph.
# Run with:  Rscript random_graph_models.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::sample_gnp(n, p)                             -- Erdős-Rényi G(n, p)\n")
  cat("  igraph::sample_gnm(n, m)                             -- Erdős-Rényi G(n, m)\n")
  cat("  igraph::sample_smallworld(1, n, k, p)                -- Watts-Strogatz\n")
  cat("  igraph::sample_pa(n, m, directed=FALSE)              -- Barabási-Albert / preferential attachment\n")
  cat("  igraph::sample_fitness(n, fitness)                   -- fitness / Chung-Lu\n")
  cat("  igraph::sample_degseq(deg)                           -- configuration model\n")
}
