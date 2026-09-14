# beta-VAE for Disentanglement (Reference Sec 47.288)
# Native R via reticulate + PyTorch / TF; Python via disentanglement-lib / from-scratch.
# Run with:  Rscript beta_vae_disentangle.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + PyTorch / TensorFlow VAE  -- fit custom beta-VAE\n")
  cat("  torch (R) with beta-adjusted KL term   -- native but sparse\n")
  cat("Python:\n")
  cat("  google-research/disentanglement_lib\n")
  cat("  Pyro / NumPyro (mini-vae examples)\n")
  cat("  From-scratch numpy (see beta_vae_disentangle.py)\n")
  cat("Refs: Higgins, I. et al (2017) 'beta-VAE: Learning basic visual concepts\n")
  cat("      with a constrained variational framework', ICLR.\n")
}
