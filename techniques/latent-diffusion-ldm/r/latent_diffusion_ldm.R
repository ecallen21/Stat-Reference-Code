# Latent Diffusion / Stable Diffusion (Reference Sec 47.211).
#
# Rombach et al 2022 CVPR. VAE-encoded latent-space diffusion;
# 64× cheaper per step than pixel-space diffusion.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Latent Diffusion / Stable Diffusion (Rombach et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * diffusers.StableDiffusionPipeline / SD3Pipeline\n")
cat("  * diffusers.AutoencoderKL (VAE)\n")
cat("  * CompVis/latent-diffusion (paper reference)\n")
