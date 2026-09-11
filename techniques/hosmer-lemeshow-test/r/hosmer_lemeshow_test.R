# Hosmer-Lemeshow Test (Reference Sec 47.256)
# Native R via ResourceSelection / rms / generalhoslem; Python via statsmodels workaround / from-scratch.
# Run with:  Rscript hosmer_lemeshow_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ResourceSelection::hoslem.test  -- classical H-L test\n")
  cat("  generalhoslem::logitgof         -- multinomial / ordinal extensions\n")
  cat("  rms::val.prob                    -- full calibration + H-L bundle\n")
  cat("  performance::performance_hosmer  -- part of easystats\n")
  cat("Python:\n")
  cat("  statsmodels (manual: deciles + chi-square approximation)\n")
  cat("  From-scratch scipy (see hosmer_lemeshow_test.py)\n")
  cat("Refs: Hosmer, D.W. & Lemeshow, S. (1980) 'Goodness-of-fit tests for the\n")
  cat("      multiple logistic regression model', CommStat A9(10), 1043-1069.\n")
}
