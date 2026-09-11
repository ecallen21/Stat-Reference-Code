# Tree of Thoughts (Reference Sec 47.184).
#
# Yao et al 2023. Explicit BFS / DFS over reasoning steps with an
# LM-based value estimator; beats single-chain CoT on planning /
# puzzle tasks.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Tree of Thoughts (Yao et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * llm-reasoners (reference frameworks)\n")
cat("  * langgraph, lmql\n")
cat("  * princeton-nlp/tree-of-thought-llm (paper authors' repo)\n")
