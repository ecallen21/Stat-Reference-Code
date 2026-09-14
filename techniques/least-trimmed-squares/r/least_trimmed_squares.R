# Least Trimmed Squares (Rousseeuw 1984 JASA)
# R: `robustbase::ltsReg`, `MASS::lqs(method='lts')`,
#    `MASS::rlm` (M-estimator)
# Python: `sklearn.linear_model.HuberRegressor`,
#         `sklearn.linear_model.RANSACRegressor` (similar
#         robust selection), from-scratch FAST-LTS
#
# library(robustbase)
# fit <- ltsReg(y ~ ., data = df, alpha = 0.75)
