# Consistency Models (Reference Sec 47.215).
#
# Song et al 2023 ICML. Self-consistency along diffusion ODE
# trajectories; enables 1-step high-quality generation.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Consistency Models (Song et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * openai/consistency_models (paper reference)\n")
cat("  * diffusers.CMStochasticIterativeScheduler\n")
cat("  * latent-consistency-model variants for text-to-image\n")
