# RAGAS (Reference Sec 47.218).
#
# Es et al 2024. Reference-free RAG metrics: faithfulness, answer
# relevance, context precision / recall.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== RAGAS (Es et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * ragas (main package)\n")
cat("  * deepeval\n")
cat("  * trulens-eval\n")
