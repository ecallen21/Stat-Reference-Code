# Hodrick-Prescott Filter (Reference Sec 47.143).
#
# Hodrick & Prescott 1997. Trend-cycle decomposition via penalised
# 2nd-difference least-squares.
#
# Uses `mFilter` for HP + other trend-cycle filters.

if (!requireNamespace("mFilter", quietly = TRUE)) install.packages("mFilter")
library(mFilter)                                      # HP, BK, CF filters

set.seed(0)
T <- 200
tt <- 0:(T - 1)
trend_true <- 0.05 * tt + 0.0003 * tt^2
cycle_true <- 2 * sin(2 * pi * tt / 20)
y <- trend_true + cycle_true + rnorm(T, sd = 0.5)

for (lambda in c(10, 1600, 129600)) {
    hp <- hpfilter(y, freq = lambda)
    rmse_t <- sqrt(mean((as.numeric(hp$trend) - trend_true)^2))
    rmse_c <- sqrt(mean((as.numeric(hp$cycle) - cycle_true)^2))
    cat(sprintf("  lambda = %6d   trend RMSE = %.3f   cycle RMSE = %.3f\n",
                lambda, rmse_t, rmse_c))
}
