# ReAct (Reference Sec 47.202).
#
# Yao et al 2023 ICLR. Interleaves Thought / Action / Observation
# to combine reasoning and tool use in one loop.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ReAct (Yao et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * langchain ReActAgent\n")
cat("  * llama-index ReActAgent (docs)\n")
cat("  * dspy ReAct module\n")
