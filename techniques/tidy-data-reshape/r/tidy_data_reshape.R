# Tidy data + long/wide reshaping (Reference Sec 41.15)
# Native R via tidyr / data.table; Python pandas + custom.
# Run with:  Rscript tidy_data_reshape.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tidyr::pivot_longer / pivot_wider  -- Hadley Wickham's tidy interface\n")
  cat("  data.table::melt / dcast           -- fast long/wide reshaping\n")
  cat("  reshape2::melt / dcast / acast     -- earlier generation, still supported\n")
  cat("Python:\n")
  cat("  pandas::melt / pivot / pivot_table / stack / unstack / wide_to_long\n")
  cat("  polars::melt / pivot                -- fast alternative\n")
  cat("Refs: Wickham, H. (2014) 'Tidy data', JSS 59(10); Wickham & Grolemund (2023)\n")
  cat("      R for Data Science, 2nd ed., O'Reilly, Ch 5-6.\n")
}
