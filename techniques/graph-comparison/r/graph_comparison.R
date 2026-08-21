# Graph comparison (Reference §24.12)
# R via graphkernels or NetworkDistance.
# Run with:  Rscript graph_comparison.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  NetworkDistance::nd.gdd / nd.dsd / nd.he / nd.hamming    -- 12+ graph distances\n")
  cat("  graphkernels::CalculateWLKernel(...)                       -- Weisfeiler-Lehman kernel\n")
  cat("  graphkernels::CalculateShortestPathKernel(...)             -- shortest-path kernel\n")
  cat("  igraph::graph.isomorphic.vf2                              -- exact iso check\n")
  cat("Python:  netrd.distance (spectral, DeltaCon, portrait divergence, ...)\n")
  cat("         gmatch4py: multiple GED approximations + kernels\n")
}
