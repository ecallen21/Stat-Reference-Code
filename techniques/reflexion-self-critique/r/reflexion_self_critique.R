# Reflexion (Reference Sec 47.201).
#
# Shinn et al 2023 NeurIPS. Episodic-memory verbal RL over LM agent
# trials; retry with reflection appended to prompt.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Reflexion (Shinn et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * noahshinn024/reflexion (reference impl)\n")
cat("  * langgraph reflection agents\n")
cat("  * llama-index self-critique modules\n")
