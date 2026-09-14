# CycleGAN (Reference Sec 47.287)
# Native R via reticulate + PyTorch; Python via junyanz/pytorch-CycleGAN / from-scratch.
# Run with:  Rscript cyclegan_unpaired_translation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + PyTorch CycleGAN         -- standard workflow\n")
  cat("  torch (R) with custom cycle-consistency loss\n")
  cat("Python:\n")
  cat("  junyanz/pytorch-CycleGAN-and-pix2pix  -- reference implementation\n")
  cat("  tf-gan cyclegan_gan_model\n")
  cat("  From-scratch numpy (see cyclegan_unpaired_translation.py)\n")
  cat("Refs: Zhu, J.-Y., Park, T., Isola, P. & Efros, A.A. (2017) 'Unpaired\n")
  cat("      image-to-image translation using cycle-consistent adversarial\n")
  cat("      networks', ICCV.\n")
}
