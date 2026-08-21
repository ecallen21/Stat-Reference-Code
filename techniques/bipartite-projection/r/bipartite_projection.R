# Bipartite projection + modularity (Reference §24.10)
# R via igraph, bipartite, or tnet.
# Run with:  Rscript bipartite_projection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::bipartite_projection(g, which='true' | 'false')  -- weighted projections\n")
  cat("  igraph::cluster_optimal(...)                              -- exact modularity for small graphs\n")
  cat("  bipartite::computeModules(web, method='DIRT_LPA_wb_plus') -- LP-Wolf bipartite mod\n")
  cat("  bipartite::PDI  bipartite::specieslevel                   -- ecological summaries\n")
  cat("  tnet::projecting_tm(net, method='Newman')                 -- Newman hyperbolic weighting\n")
}
