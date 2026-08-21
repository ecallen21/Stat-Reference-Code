# Contrastive representation learning (Reference §27.x extra)
# R via torch or reticulate + Python contrastive libraries.
# Run with:  Rscript contrastive_learning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: manual NT-Xent loss on top of nn_module encoder\n")
  cat("  reticulate + lightly / pytorch-metric-learning (Python)\n")
  cat("Modern contrastive frameworks (all Python):\n")
  cat("  SimCLR (Chen 2020)  - two aug views + NT-Xent + projector head\n")
  cat("  MoCo v3 (He 2020)    - momentum encoder queue, no massive batch\n")
  cat("  BYOL (Grill 2020)    - no negatives, target network + predictor\n")
  cat("  SimSiam (Chen 2021)  - stop-gradient + predictor; no negatives\n")
  cat("  DINO (Caron 2021)    - self-distillation with teacher / student\n")
  cat("  CLIP (Radford 2021)   - image-text contrastive; bi-encoder\n")
  cat("  SimCSE (Gao 2021)     - contrastive sentence embeddings\n")
  cat("  E5 / GTE / BGE / mxbai — modern sentence-embedding models trained contrastively.\n")
}
