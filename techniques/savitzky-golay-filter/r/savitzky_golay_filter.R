# Savitzky-Golay Filter (Reference Sec 47.142).
#
# Savitzky & Golay 1964. Sliding-window polynomial LS smoother.
#
# Uses `signal::sgolayfilt` or `prospectr::savitzkyGolay`.

if (!requireNamespace("signal", quietly = TRUE)) install.packages("signal")
library(signal)                                       # signal-processing filters

set.seed(0)
t <- seq(0, 4 * pi, length.out = 400)
clean <- sin(t) + 0.3 * sin(3 * t)
y <- clean + rnorm(length(t), sd = 0.3)

y_smooth <- sgolayfilt(y, p = 3, n = 21)
rmse_raw <- sqrt(mean((y - clean)^2))
rmse_sg  <- sqrt(mean((y_smooth - clean)^2))
cat("=== Savitzky-Golay filter (Savitzky-Golay 1964) ===\n")
cat(sprintf("  raw RMSE = %.4f   SG (w=21, p=3) RMSE = %.4f\n", rmse_raw, rmse_sg))
