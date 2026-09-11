# Tensor Parallelism / Megatron-LM (Reference Sec 47.175).
#
# Shoeybi et al 2020. Shards individual weight matrices across
# devices with two all-reduces per Transformer block.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Tensor Parallelism (Shoeybi et al 2020) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * megatron-lm (NVIDIA reference impl)\n")
cat("  * colossalai\n")
cat("  * accelerate (HF), transformers pipeline / device_map='auto'\n")
