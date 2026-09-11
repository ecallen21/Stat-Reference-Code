# Pipeline Parallelism / GPipe (Reference Sec 47.176).
#
# Huang et al 2019. Split deep model into K stages, mini-batch into
# M micro-batches, keep all K devices busy. Bubble ~ (K-1)/(M+K-1).

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Pipeline Parallelism / GPipe (Huang et al 2019) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * torch.distributed.pipeline\n")
cat("  * deepspeed pipeline (1F1B, interleaved)\n")
cat("  * megatron-lm (interleaved 1F1B)\n")
