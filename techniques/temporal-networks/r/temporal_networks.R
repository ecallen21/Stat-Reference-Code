# Temporal networks (Reference §24.x extra)
# R via timeordered, tsna, or networkDynamic.
# Run with:  Rscript temporal_networks.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  networkDynamic::networkDynamic(edge.spells=events)   -- store dynamic edges\n")
  cat("  tsna::tPath / tReach                                  -- time-respecting paths & reachability\n")
  cat("  tsna::tSnaStats                                       -- rolling network statistics\n")
  cat("  timeordered::plot.tel                                 -- ordered time-event plots\n")
  cat("  ndtv::render.d3movie                                  -- animated temporal-net visualisation\n")
  cat("Python: teneto (temporal community detection, GLS metrics); pathpy2 (higher-order network models).\n")
}
