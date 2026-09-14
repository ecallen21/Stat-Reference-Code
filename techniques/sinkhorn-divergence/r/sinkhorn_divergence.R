# Sinkhorn Divergence (Reference Sec 47.314)
# Native R via T4transport / transport / reticulate + POT; Python via POT / geomloss.
# Run with:  Rscript sinkhorn_divergence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  T4transport                     -- Sinkhorn + de-biased divergence\n")
  cat("  transport::sinkhorn            -- entropic OT\n")
  cat("  reticulate + geomloss           -- differentiable OT via PyTorch\n")
  cat("Python:\n")
  cat("  POT.sinkhorn / POT.bregman.sinkhorn_divergence\n")
  cat("  geomloss.SamplesLoss('sinkhorn')\n")
  cat("  From-scratch (see sinkhorn_divergence.py)\n")
  cat("Refs: Cuturi, M. (2013) 'Sinkhorn distances: Lightspeed computation\n")
  cat("      of optimal transport', NeurIPS;  Feydy, J. et al (2019)\n")
  cat("      'Interpolating between optimal transport and MMD using Sinkhorn\n")
  cat("      divergences', AISTATS.\n")
}
