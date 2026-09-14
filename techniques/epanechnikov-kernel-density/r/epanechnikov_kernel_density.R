# Epanechnikov kernel density (Epanechnikov 1969)
# R: `density(kernel='epanechnikov')` (base R), `KernSmooth::bkde`,
#    `ks::kde(kernel='epanechnikov')` (multi-D)
# Python: `sklearn.neighbors.KernelDensity(kernel='epanechnikov')`,
#         `statsmodels.nonparametric.kde`, from-scratch
#
# fit <- density(data, kernel = "epanechnikov", bw = "nrd0")
# plot(fit)
