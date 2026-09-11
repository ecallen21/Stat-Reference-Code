# Process Reward Model - PRM (Reference Sec 47.195).
#
# Lightman et al 2023 (OpenAI PRM800K); Wang et al 2024 Math-Shepherd.
# Score each intermediate step of a chain of thought, not just the
# final answer.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Process Reward Model (Lightman et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * openai/prm800k (dataset + reference impl)\n")
cat("  * peiyi-wang/math-shepherd (Math-Shepherd PRM)\n")
cat("  * trl RewardTrainer + PRM-style step-level labels\n")
