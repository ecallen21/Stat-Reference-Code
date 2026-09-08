# BYOL / SimSiam (Reference Sec 47.50)
# Deep-learning workflows -- Python only.
# Run with:  Rscript byol_simsiam.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  none in mainstream R -- SSL for images lives in Python\n")
  cat("Python:\n")
  cat("  lightly            -- BYOL, SimSiam, MoCo, DINO, SwAV in one package\n")
  cat("  solo-learn         -- research-focused, distributed multi-GPU\n")
  cat("  torchvision + custom -- reference implementations by paper authors\n")
  cat("  vissl              -- Facebook Research VISSL for image SSL\n")
  cat("  from-scratch       -- see byol_simsiam.py\n")
  cat("Refs: Grill et al (2020) 'BYOL' NeurIPS; Chen & He (2021) 'SimSiam' CVPR;\n")
  cat("      Caron et al (2020) 'SwAV' NeurIPS; Caron et al (2021) 'DINO' ICCV.\n")
}
