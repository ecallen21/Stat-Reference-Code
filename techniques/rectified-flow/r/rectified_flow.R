# Rectified Flow (Reference Sec 47.214).
#
# Liu-Gong-Liu 2023. Straight-line flow between prior and data;
# reflow iterations enable 1-step generation.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Rectified Flow (Liu-Gong-Liu 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * gnobitab/RectifiedFlow (paper reference)\n")
cat("  * diffusers.FlowMatchEulerDiscreteScheduler (SD3, Flux)\n")
