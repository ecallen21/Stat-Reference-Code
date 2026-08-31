# Fused LASSO / Total Variation (Reference Sec 32.13)
# Native R via genlasso; Python via skimage / cvxpy.
# Run with:  Rscript fused_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  genlasso::fusedlasso1d       -- Tibshirani-Taylor reference (path algorithm)\n")
  cat("  genlasso::trendfilter         -- higher-order fused LASSO / trend filter\n")
  cat("  changepoint                   -- classical changepoint detectors (adjacent)\n")
  cat("Python:\n")
  cat("  skimage.restoration.denoise_tv_chambolle -- Chambolle 2004 TV denoising\n")
  cat("  cvxpy + custom TV constraint            -- exact fused LASSO\n")
  cat("  ruptures                                 -- changepoint detection library\n")
  cat("  neurodsp                                  -- signal denoising\n")
  cat("Refs: Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. & Knight, K. (2005)\n")
  cat("      'Sparsity and smoothness via the fused LASSO', JRSS-B;\n")
  cat("      Chambolle, A. (2004) 'An algorithm for total variation minimization\n")
  cat("      and applications', J Math Imaging Vis.\n")
}
