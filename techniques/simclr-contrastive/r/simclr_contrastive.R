# SimCLR (Reference Sec 47.153).
#
# Chen, Kornblith, Norouzi & Hinton 2020. NT-Xent contrastive
# self-supervised learning with strong data augmentation.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== SimCLR (Chen et al 2020) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * lightly (SimCLR, MoCo, BYOL modules)\n")
cat("  * solo-learn (self-supervised zoo)\n")
cat("  * pytorch-metric-learning (NTXentLoss)\n")
