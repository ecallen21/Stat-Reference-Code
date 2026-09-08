# Affinity Propagation (Reference Sec 47.80)
# Native R via apcluster; Python via sklearn.
# Run with:  Rscript affinity_propagation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  apcluster::apcluster          -- Bodenhofer, Kothmeier & Hochreiter\n")
  cat("  apcluster::apclusterK         -- fixed-K variant via bisection on preference\n")
  cat("  apcluster::preferenceRange    -- automatic preference sweep\n")
  cat("Python:\n")
  cat("  sklearn.cluster.AffinityPropagation  -- damping + convergence checks\n")
  cat("  from-scratch                          -- see affinity_propagation.py\n")
  cat("Refs: Frey & Dueck (2007) Science 315(5814); Bodenhofer, Kothmeier &\n")
  cat("      Hochreiter (2011) 'APCluster: an R package for affinity propagation',\n")
  cat("      Bioinformatics 27(17).\n")
}
