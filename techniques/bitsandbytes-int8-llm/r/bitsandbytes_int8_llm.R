# LLM.int8() / bitsandbytes (Reference Sec 47.179).
#
# Dettmers et al 2022. Vector-wise INT8 quantisation with an
# outlier-feature FP16 fallback path.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== LLM.int8() / bitsandbytes (Dettmers et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * bitsandbytes (nn.Linear8bitLt, 4-bit NF4 in QLoRA)\n")
cat("  * transformers.load_in_8bit / load_in_4bit\n")
cat("  * bitsandbytes-foundation/bitsandbytes\n")
