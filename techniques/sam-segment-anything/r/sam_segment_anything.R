# SAM - Segment Anything (Reference Sec 47.229).
#
# Kirillov et al 2023 ICCV. Foundation segmentation model: ViT-H
# image encoder + prompt encoder + mask decoder; zero-shot masks
# from clicks / boxes / text.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== SAM (Kirillov et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * facebookresearch/segment-anything (SAM v1)\n")
cat("  * facebookresearch/sam2 (video SAM v2)\n")
cat("  * MobileSAM, EfficientSAM for edge deployment\n")
