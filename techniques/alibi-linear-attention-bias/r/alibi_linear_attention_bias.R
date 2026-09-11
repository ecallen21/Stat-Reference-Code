# ALiBi - Attention with Linear Biases (Reference Sec 47.170).
#
# Press, Smith & Lewis 2022. Fixed geometric per-head linear-in-distance
# bias replaces learned positional embeddings; extrapolates to
# sequences longer than seen at training.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ALiBi - Attention with Linear Biases (Press et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * transformers Bloom / MPT / OPT (ALiBi implementations in HF hub)\n")
cat("  * xformers / vllm ALiBi kernels\n")

# Show slopes for reference
n_heads <- 8
slopes <- 2^(-8 * seq_len(n_heads) / n_heads)
cat(sprintf("ALiBi slopes for H = %d heads: %s\n",
             n_heads, toString(round(slopes, 4))))
