# ORPO - Odds Ratio Preference Optimization (Sec 47.198).
#
# Hong-Lee-Thorne 2024. Monolithic SFT + preference loss with no
# reference model; halves memory vs DPO.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ORPO (Hong et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * trl.ORPOTrainer, trl ORPOConfig\n")
cat("  * axolotl ORPO fine-tuning template\n")
