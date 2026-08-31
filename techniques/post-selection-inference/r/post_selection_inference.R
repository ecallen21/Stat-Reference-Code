# Post-selection inference (Reference Sec 32.8)
# Native R via selectiveInference / hdi; Python via reticulate.
# Run with:  Rscript post_selection_inference.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  selectiveInference           -- Lee-Sun-Sun-Taylor conditional PoSI CIs\n")
  cat("  hdi                          -- lasso.proj, boot.lasso.proj (debiased LASSO)\n")
  cat("  PoSI                          -- Berk-Brown-Buja simultaneous PoSI\n")
  cat("Python:\n")
  cat("  selectinf                     -- unified selective inference toolkit\n")
  cat("  celer                          -- fast LASSO for the selection step\n")
  cat("Refs: Berk, R., Brown, L., Buja, A., Zhang, K. & Zhao, L. (2013) 'Valid\n")
  cat("      post-selection inference', Annals of Statistics;\n")
  cat("      Lee, J.D., Sun, D.L., Sun, Y. & Taylor, J.E. (2016) 'Exact post-selection\n")
  cat("      inference, with application to the LASSO', Annals of Statistics.\n")
}
