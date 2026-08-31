# Functional depth + outlier detection (Reference Sec 31.9)
# Native R via fda.usc / roahd; Python via scikit-fda.
# Run with:  Rscript functional_depth.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda.usc::depth.mode / depth.FM / depth.RP -- functional depth catalogue\n")
  cat("  roahd::MBD                                -- modified band depth + fBoxplot\n")
  cat("  refund                                     -- adjacent FDA regression + depth\n")
  cat("Python:\n")
  cat("  scikit-fda                                 -- ModifiedBandDepth, Fraiman-Muniz, IntegratedDepth\n")
  cat("  fdasrsf                                    -- SRSF-based depths\n")
  cat("Refs: Lopez-Pintado, S. & Romo, J. (2009) 'On the concept of depth for\n")
  cat("      functional data', JASA;\n")
  cat("      Fraiman, R. & Muniz, G. (2001) 'Trimmed means for functional data', Test.\n")
}
