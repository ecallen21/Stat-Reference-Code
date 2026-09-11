# GPTQ (Reference Sec 47.180).
#
# Frantar et al 2023. Post-training INT4/INT3 quantisation guided by
# calibration Hessian; OBS-style column-wise updates.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== GPTQ (Frantar et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * auto-gptq (community reference)\n")
cat("  * IST-DASLab/gptq (Frantar official)\n")
cat("  * optimum (HuggingFace GPTQ integration)\n")
