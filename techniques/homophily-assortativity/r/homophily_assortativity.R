# Homophily + assortativity (Reference Sec 30.17)
# Native R via igraph; Python via networkx.
# Run with:  Rscript homophily_assortativity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  igraph::assortativity_nominal  -- categorical attribute assortativity\n")
  cat("  igraph::assortativity_degree   -- degree assortativity (Newman 2002)\n")
  cat("  sna::gden, sna::nacf           -- Statnet suite assortativity helpers\n")
  cat("Python:\n")
  cat("  networkx.algorithms.assortativity.attribute_assortativity_coefficient\n")
  cat("  networkx.algorithms.assortativity.degree_assortativity_coefficient\n")
  cat("  graph-tool.correlations                          -- fast large-scale\n")
  cat("Refs: Newman, M.E.J. (2003) 'Mixing patterns in networks', Phys Rev E;\n")
  cat("      Newman, M.E.J. (2002) 'Assortative mixing in networks', Phys Rev Lett.\n")
}
