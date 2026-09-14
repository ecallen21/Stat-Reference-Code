# Croston / SBC intermittent-demand forecasting
# R: `forecast::croston` (Croston 1972 + SBC option),
#    `tsintermittent::crost` (Croston + variants),
#    `tsintermittent::tsb` (Teunter-Syntetos-Babai).
# Python: `statsforecast.IMAPA`, `sktime.CrostonForecaster`,
#         from-scratch
#
# library(forecast)
# y <- c(0, 0, 3, 0, 0, 0, 5, 0, 0, 4)
# croston(y, h = 6)
