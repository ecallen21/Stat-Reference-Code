# 3D Gaussian Splatting (Reference Sec 47.224).
#
# Kerbl et al 2023 SIGGRAPH. Millions of anisotropic 3D Gaussians +
# differentiable rasterisation for real-time novel view synthesis.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== 3D Gaussian Splatting (Kerbl et al 2023) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * graphdeco-inria/gaussian-splatting (paper reference)\n")
cat("  * nerfstudio splatfacto\n")
cat("  * gsplat (differentiable CUDA rasteriser)\n")
