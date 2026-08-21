# Spatial autoregressive models (Reference §23.9)
# R via spatialreg (Bivand).
# Run with:  Rscript spatial_autoregressive_sar.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spatialreg::lagsarlm(y ~ X, listw)    -- SAR lag MLE\n")
  cat("  spatialreg::errorsarlm(y ~ X, listw)  -- SAR error MLE\n")
  cat("  spatialreg::sacsarlm                  -- combined SAC (both terms)\n")
}
