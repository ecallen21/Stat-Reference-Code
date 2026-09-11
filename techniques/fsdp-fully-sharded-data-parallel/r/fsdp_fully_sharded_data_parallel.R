# FSDP - Fully Sharded Data Parallel (Reference Sec 47.177).
#
# Zhao et al 2023. Shards params + grads + optim states across N
# data-parallel ranks; all-gathers just-in-time for compute.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== FSDP (Zhao et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * torch.distributed.fsdp (native PyTorch)\n")
cat("  * accelerate (HuggingFace wrapper)\n")
cat("  * deepspeed ZeRO stage 3 (equivalent)\n")
