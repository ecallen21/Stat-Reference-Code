# Functional time series forecasting (Reference §13.x extra)
# R via ftsa (Rob Hyndman).
# Run with:  Rscript functional_time_series.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ftsa::fts(x = ages, y = mort_matrix)                 -- build a functional time series\n")
  cat("  ftsa::ftsm(fts_obj, order = K)                        -- Hyndman-Ullah FPCA + AR model\n")
  cat("  forecast(ftsm_obj, h = 10)                           -- h-step-ahead functional forecast\n")
  cat("  ftsa::rwd, ftsa::rar                                  -- robust variants\n")
  cat("  demography::forecast.demogdata / lca                  -- Lee-Carter for mortality data\n")
  cat("  fda::pca.fd                                           -- classical FPCA on smoothed curves\n")
  cat("Python: skfda.preprocessing.dim_reduction.FPCA + statsmodels VAR / VARIMA on scores.\n")
}
