# LLM-as-a-Judge (Reference Sec 47.216).
#
# Zheng et al 2023. Automated LM-based evaluation for open-ended
# text: single-answer grading, pairwise, reference-based.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== LLM-as-a-Judge (Zheng et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * MT-Bench, chatbot-arena-leaderboard\n")
cat("  * alpaca-eval / tatsu-lab (auto-eval)\n")
cat("  * deepeval, promptfoo, ragas\n")
