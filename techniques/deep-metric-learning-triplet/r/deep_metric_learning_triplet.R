# Deep Metric Learning with Triplet Loss (Reference Sec 47.151).
#
# Schroff, Kalenichenko & Philbin 2015 (FaceNet). Triplet loss
# with semi-hard mining for supervised embedding.
#
# No native R triplet library; use Python via reticulate.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Triplet loss embedding (Schroff et al 2015) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pytorch-metric-learning (TripletMarginLoss, semi-hard mining)\n")
cat("  * tensorflow-similarity\n")
cat("  * keras.losses.TripletSemiHardLoss\n")
