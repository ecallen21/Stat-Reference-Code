# Rectified Adam - RAdam (Reference Sec 47.165).
#
# Liu et al 2020. Rectified adaptive learning rate variance;
# falls back to SGD when the variance estimate is unreliable.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Rectified Adam - RAdam (Liu et al 2020) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pytorch_optimizer.RAdam\n")
cat("  * torch.optim.RAdam (built-in as of PyTorch 1.5+)\n")
cat("  * timm's optim factory (radam / ranger)\n")
