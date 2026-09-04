# Quantitative content analysis (Reference Sec 42.17)
# Native R via irr / irrCAC / quanteda; Python krippendorff + custom.
# Run with:  Rscript content_analysis_coding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  irr::kappa2                     -- Cohen kappa\n")
  cat("  irr::kripp.alpha                -- Krippendorff alpha (any scale)\n")
  cat("  irrCAC::gwet.ac1                -- paradox-resistant chance correction\n")
  cat("  quanteda                        -- dfm + dictionary + coding framework\n")
  cat("Python:\n")
  cat("  krippendorff                    -- Krippendorff alpha\n")
  cat("  sklearn.metrics.cohen_kappa_score\n")
  cat("Refs: Krippendorff (2019) Content Analysis: An Introduction to Its Methodology,\n")
  cat("      4th ed., SAGE; Neuendorf (2017) The Content Analysis Guidebook, 2nd ed.,\n")
  cat("      SAGE.\n")
}
