# Generative Adversarial Network (Reference §27.9)
# R via torch or keras3.
# Run with:  Rscript gan_training.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_module for Generator and Discriminator; torch::nnf_binary_cross_entropy_with_logits\n")
  cat("  keras3 with two Sequential models and custom train_step\n")
  cat("Python:\n")
  cat("  torch nn.Sequential Generator + Discriminator + torch.optim.Adam betas=(0.5, 0.999)\n")
  cat("  Wasserstein-GAN (WGAN-GP): dropout the sigmoid, penalise gradient norm.\n")
  cat("  StyleGAN, BigGAN, GAN progressive-growth — production image GANs.\n")
  cat("  Modern generative modelling shifted to diffusion / flow models (better sample quality).\n")
}
