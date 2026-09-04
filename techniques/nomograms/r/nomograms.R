# Nomograms (Reference Sec 39.3)
# Native R via rms::nomogram; Python custom via matplotlib.
# Run with:  Rscript nomograms.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms::nomogram + plot.nomogram  -- canonical nomogram from lrm / cph\n")
  cat("  regplot                        -- enhanced nomogram plots\n")
  cat("  hdnom                          -- nomograms for high-dim models (LASSO / EN)\n")
  cat("Python:\n")
  cat("  custom matplotlib              -- nomogram plot from coefficients\n")
  cat("  pynomo                         -- engineering-style nomograms\n")
  cat("Refs: Iasonos, Schrag, Raj & Panageas (2008) 'How to build and interpret a\n")
  cat("      nomogram for cancer prognosis', JCO; Harrell, F.E. (2015) Regression\n")
  cat("      Modeling Strategies, 2nd ed., Springer, Ch 14.\n")
}
