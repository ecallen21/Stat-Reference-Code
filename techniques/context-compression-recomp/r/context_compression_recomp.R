# Context Compression / RECOMP (Reference Sec 47.193).
#
# Xu et al 2024 ICLR. Extractive or abstractive compression of
# retrieved passages before feeding into the LLM.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Context Compression / RECOMP (Xu et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * carriex/recomp (paper reference impl)\n")
cat("  * llmlingua (Microsoft, prompt compression)\n")
cat("  * langchain LLMChainExtractor, ContextualCompressionRetriever\n")
