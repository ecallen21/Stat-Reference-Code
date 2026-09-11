# DreamBooth (Reference Sec 47.222).
#
# Ruiz et al 2023 CVPR. Subject-driven fine-tuning of a text-to-image
# diffusion model with prior-preservation regularisation.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== DreamBooth (Ruiz et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * diffusers train_dreambooth.py + LoRA extension\n")
cat("  * kohya-ss/sd-scripts (production DreamBooth trainer)\n")
cat("  * XavierXiao/Dreambooth-Stable-Diffusion (reference)\n")
