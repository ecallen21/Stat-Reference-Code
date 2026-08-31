# Stability selection (Reference Sec 32.6)
# Native R via stabs; Python via stability-selection package.
# Run with:  Rscript stability_selection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stabs                        -- Meinshausen-Buhlmann + complementary pairs (Shah-Samworth)\n")
  cat("  bootLASSO                    -- bootstrap-based LASSO stability\n")
  cat("Python:\n")
  cat("  stability-selection          -- pip package with the same interface\n")
  cat("  sklearn.utils.resample + LASSO -- manual reimplementation\n")
  cat("Refs: Meinshausen, N. & Buhlmann, P. (2010) 'Stability selection', JRSS-B;\n")
  cat("      Shah, R.D. & Samworth, R.J. (2013) 'Variable selection with error\n")
  cat("      control: another look at stability selection', JRSS-B.\n")
}
