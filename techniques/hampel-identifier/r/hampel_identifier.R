# Hampel identifier (Hampel 1971)
# R: `pracma::hampel`, `imputeTS::na_seasplit` (no direct
#    Hampel), `robustbase::mc` (medcouple, related);
#    time-series version in `seewave::hampel_filter`.
# Python: `pandas` rolling median-abs-deviation (custom),
#         `scipy.signal.medfilt` (median filter cousin),
#         from-scratch
#
# library(pracma)
# x <- c(rnorm(200), 15, -20, 25)
# fit <- hampel(x, k = 3, t0 = 3)
# fit$ind  # indices of detected outliers
