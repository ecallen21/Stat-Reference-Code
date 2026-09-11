# Gradient Checkpointing (Reference Sec 47.169).
#
# Chen et al 2016 'Training Deep Nets with Sublinear Memory Cost'.
# O(sqrt(L)) memory instead of O(L) for L-layer nets by recomputing
# activations during backward.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Gradient Checkpointing (Chen et al 2016) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * torch.utils.checkpoint.checkpoint (wraps a submodule)\n")
cat("  * transformers.PreTrainedModel.gradient_checkpointing_enable()\n")
cat("  * jax.checkpoint / flax.core.checkpoint\n")
