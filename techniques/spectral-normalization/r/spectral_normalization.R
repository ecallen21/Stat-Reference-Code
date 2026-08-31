# Spectral normalisation (Reference Ch 30 Robustness)
# R via reticulate + Python; native R can call svd() but not as cheaply.
# Run with:  Rscript spectral_normalization.R

if (sys.nframe() == 0) {
  cat("R packages: manual power iteration is trivial in R; no first-class R layer wrapper.\n")
  cat("  base svd() / RSpectra::svds  -- top singular value for the offline SN cap\n")
  cat("  torch (R port)               -- nn_utils_spectral_norm equivalent (community)\n")
  cat("Python:\n")
  cat("  torch.nn.utils.parametrize.register_parametrization + SpectralNorm\n")
  cat("  tensorflow_addons.layers.SpectralNormalization\n")
  cat("  jax / flax                   -- manual per-layer SN in flax.linen.Module\n")
  cat("Refs: Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. (2018)\n")
  cat("      'Spectral Normalization for Generative Adversarial Networks', ICLR.\n")
}
