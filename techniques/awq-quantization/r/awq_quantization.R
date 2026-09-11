# AWQ (Reference Sec 47.181).
#
# Lin et al 2024. Activation-aware INT4 weight quantisation: scale
# up 'salient' weight columns so quantisation grid resolution is
# not wasted on outliers.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== AWQ (Lin et al 2024) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * llm-awq (Han lab reference impl)\n")
cat("  * autoawq (community wrapper)\n")
cat("  * vllm.awq / awq_marlin kernels\n")
