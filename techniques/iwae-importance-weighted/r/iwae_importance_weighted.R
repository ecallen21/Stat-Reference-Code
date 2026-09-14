# IWAE (Reference Sec 47.289)
# Native R via reticulate + Pyro; Python via Pyro / TFP / from-scratch.
# Run with:  Rscript iwae_importance_weighted.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + Pyro / NumPyro   -- IWAE ELBO estimators\n")
  cat("  torch (R) with logsumexp of K IS weights\n")
  cat("Python:\n")
  cat("  pyro.infer.RenyiELBO (K-sample IWAE-style)\n")
  cat("  tfp.vi.monte_carlo_variational_loss(num_samples=K)\n")
  cat("  From-scratch scipy (see iwae_importance_weighted.py)\n")
  cat("Refs: Burda, Y., Grosse, R.B. & Salakhutdinov, R.R. (2016)\n")
  cat("      'Importance weighted autoencoders', ICLR.\n")
}
