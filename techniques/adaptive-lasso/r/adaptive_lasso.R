# Adaptive LASSO (Reference Sec 32.12)
# Native R via glmnet penalty.factor; Python via sklearn Lasso weighting.
# Run with:  Rscript adaptive_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet(penalty.factor = 1 / abs(init)^gamma)   -- adaptive LASSO recipe\n")
  cat("  ncvreg                        -- adjacent nonconvex penalties\n")
  cat("  parcor                        -- adaptive-LASSO based partial correlations\n")
  cat("Python:\n")
  cat("  celer / sklearn Lasso        -- weighted-lasso wrapper\n")
  cat("  hdlasso                       -- fast high-dim implementations\n")
  cat("Refs: Zou, H. (2006) 'The adaptive LASSO and its oracle properties', JASA.\n")
}
