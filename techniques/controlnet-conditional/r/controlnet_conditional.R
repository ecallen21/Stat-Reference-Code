# ControlNet (Reference Sec 47.213).
#
# Zhang-Rao-Agrawala 2023 ICCV. Trainable encoder branch with
# zero-init output convs added to a frozen base diffusion UNet.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== ControlNet (Zhang et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * diffusers.ControlNetModel + StableDiffusionControlNetPipeline\n")
cat("  * lllyasviel/ControlNet (paper reference)\n")
cat("  * T2I-Adapter / IP-Adapter alternatives\n")
