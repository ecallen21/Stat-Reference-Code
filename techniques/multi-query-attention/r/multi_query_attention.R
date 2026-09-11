# Multi-Query / Grouped-Query Attention (Reference Sec 47.186).
#
# Shazeer 2019 MQA; Ainslie 2023 GQA. Shared K/V across heads or
# groups for cheap KV-cache in autoregressive LLM inference.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== MQA / GQA (Shazeer 2019; Ainslie 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * transformers LlamaAttention (num_key_value_heads controls GQA groups)\n")
cat("  * vllm / TensorRT-LLM kernels\n")
