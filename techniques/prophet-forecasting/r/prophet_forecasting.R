# Prophet forecasting (Reference Sec 47.116)
# Native R via prophet / fable; Python via prophet / neuralprophet.
# Run with:  Rscript prophet_forecasting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  prophet                       -- Facebook reference package\n")
  cat("  fable::PROPHET                -- tidy-forecast wrapper\n")
  cat("  forecast::stl / ets / auto.arima  -- classical benchmarks\n")
  cat("Python:\n")
  cat("  prophet                       -- Facebook reference (Stan backend)\n")
  cat("  neuralprophet                 -- Prophet + AR-Net neural extension\n")
  cat("  darts.models.Prophet          -- unified TS framework\n")
  cat("  from-scratch                  -- see prophet_forecasting.py\n")
  cat("Refs: Taylor & Letham (2018) 'Forecasting at scale', The American\n")
  cat("      Statistician 72(1).\n")
}
