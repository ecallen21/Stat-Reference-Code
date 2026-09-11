# Cross-Encoder Reranker (Reference Sec 47.190).
#
# Nogueira-Cho 2019. Feed [q; SEP; p] through a single BERT for
# a pointwise / pairwise relevance score.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Cross-Encoder Reranker (Nogueira-Cho 2019) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * sentence-transformers.CrossEncoder\n")
cat("  * MonoT5 / MiniLM cross-encoders on HF Hub\n")
cat("  * pyserini reranker harness\n")
