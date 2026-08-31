# Selective prediction / abstention (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python; native R implementation is straightforward.
# Run with:  Rscript selective_prediction.R

if (sys.nframe() == 0) {
  cat("R packages: any confidence score + threshold suffices.\n")
  cat("  yardstick / MLmetrics       -- accuracy / calibration metrics at coverage=100%\n")
  cat("  probably                    -- tidymodels: cal_estimate_* + threshold_perf_* helpers\n")
  cat("Python:\n")
  cat("  torchsel / selective_classification -- deep learning selective heads\n")
  cat("  cleanlab                    -- confidence-based error / abstention scoring\n")
  cat("  MAPIE                       -- conformal set size = 1 as an abstention criterion\n")
  cat("Refs: Chow, C.K. (1957) 'An optimum character recognition system using decision\n")
  cat("      functions', IRE; El-Yaniv & Wiener (2010) 'On the foundations of noise-\n")
  cat("      free selective classification', JMLR; Geifman & El-Yaniv (2017)\n")
  cat("      'Selective classification for deep neural networks', NeurIPS.\n")
}
