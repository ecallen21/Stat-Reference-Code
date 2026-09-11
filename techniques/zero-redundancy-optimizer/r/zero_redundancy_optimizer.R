# ZeRO - Zero Redundancy Optimizer (Reference Sec 47.178).
#
# Rajbhandari et al 2020 DeepSpeed. Three stages of sharding
# (optim, +grads, +params); +offload for trillion-parameter models.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ZeRO / DeepSpeed (Rajbhandari et al 2020) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * deepspeed (Microsoft reference impl)\n")
cat("  * torch.distributed.fsdp (~ZeRO-3)\n")
cat("  * accelerate (unifies FSDP + DeepSpeed configs)\n")
