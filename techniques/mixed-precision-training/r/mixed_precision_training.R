# Mixed-Precision Training (Reference Sec 47.168).
#
# Micikevicius et al 2018. FP16 (or BF16) forward / backward with
# FP32 master weights + dynamic loss scaling. 2-4x speedup on
# tensor-core GPUs at no accuracy cost on most models.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Mixed-Precision Training (Micikevicius et al 2018) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * torch.amp (autocast + GradScaler)\n")
cat("  * tensorflow.keras.mixed_precision.set_global_policy('mixed_float16')\n")
cat("  * apex.amp (deprecated) / JAX bfloat16\n")
