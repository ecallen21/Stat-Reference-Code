# Neural collaborative filtering (Reference Sec 47.119)
# Python via recommenders / torch.
# Run with:  Rscript neural_collaborative_filtering.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R)                       -- build NCF manually with hooks\n")
  cat("  keras (R)                       -- Keras NCF via reticulate\n")
  cat("Python:\n")
  cat("  recommenders (Microsoft)        -- NCF, LightGCN, xDeepFM, DeepFM\n")
  cat("  neural_collaborative_filtering  -- author's reference PyTorch code\n")
  cat("  torch-rechub                    -- production-oriented recsys library\n")
  cat("  from-scratch                    -- see neural_collaborative_filtering.py\n")
  cat("Refs: He, Liao, Zhang, Nie, Hu & Chua (2017) 'Neural collaborative\n")
  cat("      filtering', WWW.\n")
}
