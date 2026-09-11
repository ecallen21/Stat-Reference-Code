# Sliding-Window Attention (Reference Sec 47.187).
#
# Beltagy 2020 Longformer; Jiang 2023 Mistral. Local-window
# causal attention with optional 'sink' tokens for streaming.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Sliding-Window Attention (Beltagy 2020; Jiang 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * transformers Mistral, Longformer, BigBird\n")
cat("  * flash-attention sliding-window kernels\n")
cat("  * StreamingLLM (attention sinks) - Xiao 2024\n")
