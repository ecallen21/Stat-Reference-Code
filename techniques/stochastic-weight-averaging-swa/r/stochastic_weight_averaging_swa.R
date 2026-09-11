# Stochastic Weight Averaging - SWA (Reference Sec 47.167).
#
# Izmailov et al 2018. Simple averaging of SGD iterates after warm-
# up; finds wider optima with better generalisation than the raw
# SGD endpoint.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Stochastic Weight Averaging (Izmailov et al 2018) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * torch.optim.swa_utils.AveragedModel + SWALR\n")
cat("  * pytorch-lightning StochasticWeightAveraging callback\n")
cat("  * timm's optim SWA wrappers\n")
