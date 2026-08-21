# Denoising Diffusion Probabilistic Model (Reference §27.x extra)
# R via torch + custom denoiser, or reticulate + Python diffusers.
# Run with:  Rscript diffusion_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual DDPM (linear beta schedule + U-Net denoiser + reparameterisation loss)\n")
  cat("  reticulate + huggingface diffusers (DDPMScheduler + UNet2DModel + StableDiffusionPipeline)\n")
  cat("  reticulate + denoising-diffusion-pytorch (Katherine Crowson / lucidrains)\n")
  cat("Variants:\n")
  cat("  DDIM (Song 2020)         -- deterministic sampler, 10-50 steps\n")
  cat("  score-based / EDM (Karras 2022)  -- SDE view; state-of-the-art image samples\n")
  cat("  latent diffusion (Rombach 2022)  -- diffuse in a VAE latent; Stable Diffusion\n")
  cat("  flow-matching (Lipman 2023)      -- straight-line probability paths; simpler and faster\n")
  cat("  consistency models (Song 2023)    -- single-step samplers via distillation\n")
}
