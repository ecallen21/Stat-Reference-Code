# Variational autoencoder (Reference §27.8)
# R via torch or keras3.
# Run with:  Rscript variational_autoencoder.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_module with encoder (nn_linear x 2 -> nn_linear x 2 for mu, log_sig2)\n")
  cat("  keras3 Sequential encoder + Sampling layer + decoder; add KL loss via add_loss()\n")
  cat("Python:\n")
  cat("  torch: manually reparameterise z = mu + sigma * eps  and combine recon + KL loss.\n")
  cat("  pyro/numpyro: pyro.sample('z', dist.Normal(mu, sig)) + pyro.plate + SVI.\n")
  cat("  tensorflow-probability: tfd.Normal, tfp.vi.monte_carlo_variational_loss.\n")
  cat("Extensions: beta-VAE (KL scaled by beta), Info-VAE, Vector-Quantised VAE (VQ-VAE),\n")
  cat("            normalising-flow priors, diffusion-based decoders.\n")
}
