# Holt-Winters (Reference Sec 47.309)
# Native R via forecast / stats; Python via statsmodels / from-scratch.
# Run with:  Rscript holt_winters_forecasting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  forecast::hw                  -- Holt-Winters (additive / multiplicative)\n")
  cat("  stats::HoltWinters            -- base-R workhorse\n")
  cat("  fable::ETS                    -- state-space ETS family\n")
  cat("  smooth::es                    -- exponential smoothing framework\n")
  cat("Python:\n")
  cat("  statsmodels.tsa.holtwinters.ExponentialSmoothing\n")
  cat("  darts.models.ExponentialSmoothing\n")
  cat("  From-scratch (see holt_winters_forecasting.py)\n")
  cat("Refs: Holt, C.C. (1957) 'Forecasting seasonals and trends by\n")
  cat("      exponentially weighted moving averages', Carnegie IIT Mem;\n")
  cat("      Winters, P.R. (1960) 'Forecasting sales by exponentially\n")
  cat("      weighted moving averages', Manag Sci 6.\n")
}
