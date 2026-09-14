# WGAN-GP (Reference Sec 47.286)
# Native R via reticulate + PyTorch; Python via torch / from-scratch.
# Run with:  Rscript wgan_gp_gradient_penalty.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + PyTorch WGAN-GP     -- standard route for GAN training in R\n")
  cat("  torch (R) with manual gradient penalty\n")
  cat("Python:\n")
  cat("  martinarjovsky/WassersteinGAN (WGAN)\n")
  cat("  igul222/improved_wgan_training (WGAN-GP original)\n")
  cat("  tf-gan Keras.WGAN + gradient_penalty\n")
  cat("  StudioGAN (POSTECH)\n")
  cat("  From-scratch numpy (see wgan_gp_gradient_penalty.py)\n")
  cat("Refs: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V. &\n")
  cat("      Courville, A.C. (2017) 'Improved training of Wasserstein GANs',\n")
  cat("      NeurIPS.\n")
}
