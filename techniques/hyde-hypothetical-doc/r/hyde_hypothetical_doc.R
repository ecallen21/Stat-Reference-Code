# HyDE - Hypothetical Document Embeddings (Reference Sec 47.191).
#
# Gao et al 2022. Generate a hypothetical passage from the query,
# then embed it (not the query) for retrieval.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== HyDE (Gao et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * llama-index HyDEQueryTransform\n")
cat("  * langchain HyDE community integration\n")
cat("  * texttron/hyde reference impl\n")
