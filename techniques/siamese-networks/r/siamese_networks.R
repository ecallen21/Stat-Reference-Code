# Siamese Networks (Reference Sec 47.152).
#
# Bromley 1994; Koch et al 2015. Twin networks with shared weights
# for pair similarity / one-shot classification.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Siamese networks (Bromley 1994; Koch et al 2015) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * keras (Siamese examples via shared sub-networks)\n")
cat("  * pytorch-metric-learning (ContrastiveLoss)\n")
cat("  * tensorflow-similarity\n")
