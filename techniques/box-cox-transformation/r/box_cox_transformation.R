# Box-Cox transformation (Reference Sec 41.1)
# Native R via MASS::boxcox; Python scipy + custom.
# Run with:  Rscript box_cox_transformation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MASS::boxcox                    -- profile log-likelihood + optimum lambda\n")
  cat("  car::powerTransform             -- Box-Cox + Yeo-Johnson + Box-Cox-Cole variants\n")
  cat("  bestNormalize                   -- auto-choose among power transforms\n")
  cat("Python:\n")
  cat("  scipy.stats.boxcox              -- MLE lambda + transformed y\n")
  cat("  sklearn.preprocessing.PowerTransformer (method='box-cox')\n")
  cat("Refs: Box, G.E.P. & Cox, D.R. (1964) 'An analysis of transformations', JRSS-B.\n")
}
