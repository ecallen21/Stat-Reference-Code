# Hausman test (Reference Sec 35.2)
# Native R via plm::phtest; Python via linearmodels.
# Run with:  Rscript hausman_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  plm::phtest                  -- Hausman FE vs RE for panel data\n")
  cat("  plm::pwtest, pcdtest         -- Wooldridge, Pesaran CD tests (adjacent)\n")
  cat("Python:\n")
  cat("  linearmodels.iv.PanelOLS + auxiliary Wooldridge regression\n")
  cat("  statsmodels                  -- MixedLM (RE) + FE via demeaning\n")
  cat("Refs: Hausman, J.A. (1978) 'Specification tests in econometrics', Econometrica;\n")
  cat("      Wooldridge, J.M. (2002) 'Econometric Analysis of Cross Section and Panel\n")
  cat("      Data', MIT Press, Ch. 10.\n")
}
