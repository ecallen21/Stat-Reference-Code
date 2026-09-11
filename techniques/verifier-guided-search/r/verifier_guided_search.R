# Verifier-Guided Search (Reference Sec 47.205).
#
# Uesato 2022; Lightman 2023; Wang 2024 Math-Shepherd. Prune bad
# reasoning paths in beam / MCTS using a learned verifier.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Verifier-Guided Search (Uesato 2022; Lightman 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * llm-reasoners (search + verifier framework)\n")
cat("  * openai/prm800k + custom beam / MCTS driver\n")
cat("  * princeton-nlp/tree-of-thought-llm\n")
