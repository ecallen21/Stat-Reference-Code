# KV-Cache Quantization (Reference Sec 47.174).
#
# Sheng et al 2023 FlexGen; Liu et al 2023 KIVI. Quantise the K, V
# cache to INT8 / INT4 for cheap LLM inference.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== KV-Cache Quantization (Sheng 2023; Liu 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * vllm (INT8/INT4 KV-cache modes)\n")
cat("  * FlexGen (Sheng et al ref impl)\n")
cat("  * llama.cpp Q8_0 / Q4_0 K-Q-V cache options\n")
