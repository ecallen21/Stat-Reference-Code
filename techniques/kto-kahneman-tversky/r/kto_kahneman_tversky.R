# KTO - Kahneman-Tversky Optimization (Sec 47.197).
#
# Ethayarajh et al 2024. Unpaired preference labels + prospect-theory
# value function; alternative to DPO that needs paired data.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== KTO (Ethayarajh et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * trl.KTOTrainer, trl KTOConfig\n")
cat("  * ContextualAI/HALOs (paper reference)\n")
