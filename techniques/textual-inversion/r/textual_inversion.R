# Textual Inversion (Reference Sec 47.223).
#
# Gal et al 2022 ICLR. Learn a single token embedding for a subject
# with the diffusion backbone frozen; ~1000x cheaper than DreamBooth.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Textual Inversion (Gal et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * diffusers textual_inversion.py training script\n")
cat("  * automatic1111 built-in TI trainer\n")
cat("  * kohya-ss / civit.ai TI ecosystem\n")
