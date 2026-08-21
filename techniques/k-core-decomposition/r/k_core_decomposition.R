# k-core decomposition and k-truss (Reference §24.x extra)
# R via igraph.
# Run with:  Rscript k_core_decomposition.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::coreness(g)              -- Batagelj-Zaversnik core number\n")
  cat("  igraph::k_core(g, min_cores=k)   -- extract the k-core subgraph\n")
  cat("  igraph::max_cores(g)             -- degeneracy (max coreness)\n")
  cat("Python: networkx.core_number, networkx.k_core, networkx.k_truss (edge-based).\n")
}
