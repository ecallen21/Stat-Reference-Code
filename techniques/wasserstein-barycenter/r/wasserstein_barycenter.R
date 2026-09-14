# Wasserstein Barycenter (Reference Sec 47.313)
# Native R via T4transport / transport; Python via POT / from-scratch.
# Run with:  Rscript wasserstein_barycenter.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  T4transport                -- barycenters, fused-Gromov, sliced OT\n")
  cat("  transport                  -- OT / Wasserstein distances (2-marg)\n")
  cat("Python:\n")
  cat("  POT (Python Optimal Transport)  -- bregman.barycenter, free-support\n")
  cat("  geomloss.SamplesLoss       -- gradient-friendly OT / Sinkhorn\n")
  cat("  From-scratch (see wasserstein_barycenter.py)\n")
  cat("Refs: Agueh, M. & Carlier, G. (2011) 'Barycenters in the Wasserstein\n")
  cat("      space', SIAM J Math Anal 43;  Cuturi, M. & Doucet, A. (2014)\n")
  cat("      'Fast computation of Wasserstein barycenters', ICML.\n")
}
