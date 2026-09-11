# Sharpness-Aware Minimization - SAM (Reference Sec 47.163).
#
# Foret et al 2021. Minimises loss in a rho-ball around current
# parameters, biasing training toward flat minima.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Sharpness-Aware Minimization (Foret et al 2021) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * sam-optimizer (davda54/sam)\n")
cat("  * torch_optimizer (jettify)\n")
cat("  * transformers Trainer with SAMPolicy\n")
