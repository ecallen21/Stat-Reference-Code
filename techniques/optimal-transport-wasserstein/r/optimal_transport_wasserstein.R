# Optimal transport / Wasserstein distance (Reference Sec 46.16)
# Native R via transport; Python via POT / geomloss.
# Run with:  Rscript optimal_transport_wasserstein.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  transport::wasserstein          -- W_p, plus emd() network-flow solver\n")
  cat("  transport::sinkhornTransport    -- entropic OT\n")
  cat("  approxOT                        -- fast Sinkhorn variants\n")
  cat("Python:\n")
  cat("  POT (Python Optimal Transport)  -- reference package\n")
  cat("  geomloss                         -- differentiable Sinkhorn on GPU (torch)\n")
  cat("  scipy.stats.wasserstein_distance -- 1-D W_1\n")
  cat("  ott-jax                         -- JAX-native OT primitives\n")
  cat("Refs: Monge, G. (1781) Memoire sur la theorie des deblais et des remblais;\n")
  cat("      Kantorovich, L.V. (1942) 'On the translocation of masses', Dokl Akad\n")
  cat("      Nauk SSSR 37; Cuturi, M. (2013) 'Sinkhorn distances', NIPS.\n")
}
