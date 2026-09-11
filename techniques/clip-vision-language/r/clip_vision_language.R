# CLIP (Reference Sec 47.226).
#
# Radford et al 2021 ICML. Joint image / text encoder trained on
# 400M pairs with in-batch symmetric contrastive loss.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== CLIP (Radford et al 2021) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * openai/CLIP (paper reference)\n")
cat("  * mlfoundations/open_clip (many variants + checkpoints)\n")
cat("  * sentence-transformers (CLIP-ViT-* models)\n")
