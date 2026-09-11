# Lookahead Optimizer (Reference Sec 47.164).
#
# Zhang, Lucas, Ba & Hinton 2019. Slow-weight EMA of a fast inner
# SGD optimizer; k steps forward, 1 step back.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Lookahead Optimizer (Zhang-Lucas-Ba-Hinton 2019) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * pytorch_optimizer.Lookahead (wraps any base optimizer)\n")
cat("  * RangerOptimizer = RAdam + Lookahead\n")
