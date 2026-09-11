# Classifier-Free Guidance (Reference Sec 47.210).
#
# Ho-Salimans 2021. Joint conditional / unconditional score training;
# guided sampling by linear combination.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== CFG (Ho-Salimans 2021) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * diffusers.StableDiffusionPipeline (guidance_scale)\n")
cat("  * openai/glide (classifier-free variant)\n")
