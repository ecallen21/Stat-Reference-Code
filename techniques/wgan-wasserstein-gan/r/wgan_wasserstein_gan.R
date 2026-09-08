# Wasserstein GAN (Reference Sec 47.109)
# Deep learning; Python for real implementations.
# Run with:  Rscript wgan_wasserstein_gan.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch                       -- R bindings; write WGAN loop manually\n")
  cat("  transport / T4transport     -- exact Wasserstein between empirical measures\n")
  cat("Python:\n")
  cat("  torchGAN / pytorch-lightning-bolts -- WGAN + WGAN-GP + spectral norm\n")
  cat("  pot                          -- Python Optimal Transport (Wasserstein solvers)\n")
  cat("  diffusers                    -- HF diffusion (successor to WGAN in many tasks)\n")
  cat("  from-scratch                 -- see wgan_wasserstein_gan.py\n")
  cat("Refs: Arjovsky, Chintala & Bottou (2017) ICML; Gulrajani et al (2017)\n")
  cat("      'Improved training of Wasserstein GANs', NeurIPS.\n")
}
