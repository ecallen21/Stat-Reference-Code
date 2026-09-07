# JEPA -- Joint Embedding Predictive Architecture (Reference Sec 47.23)
# Deep self-supervised learning; no R implementations. Python via Meta I-JEPA / V-JEPA.
# Run with:  Rscript jepa_self_supervised.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for JEPA.\n")
  cat("Python:\n")
  cat("  Meta AI I-JEPA repo (github.com/facebookresearch/ijepa)\n")
  cat("  Meta AI V-JEPA repo (github.com/facebookresearch/jepa) -- video variant\n")
  cat("  vicreg / dinov2 -- related SSL methods with anti-collapse regularisation\n")
  cat("  timm / torchvision -- backbones commonly used with JEPA\n")
  cat("Refs: LeCun, Y. (2022) 'A path towards autonomous machine intelligence',\n")
  cat("      Meta AI open review; Assran, M. et al. (2023) 'Self-supervised\n")
  cat("      learning from images with a joint-embedding predictive architecture'\n")
  cat("      (I-JEPA), CVPR; Bardes, A. et al. (2024) 'Revisiting feature\n")
  cat("      prediction for learning visual representations from video' (V-JEPA).\n")
}
