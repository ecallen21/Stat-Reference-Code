# ColBERT (Reference Sec 47.189).
#
# Khattab & Zaharia 2020. Per-token embeddings + MaxSim late-
# interaction for high-recall retrieval.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ColBERT (Khattab-Zaharia 2020) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * stanford-futuredata/ColBERT (reference impl)\n")
cat("  * bclavie/ragatouille (higher-level ColBERT wrapper)\n")
cat("  * pyserini (evaluation harness with ColBERT)\n")
