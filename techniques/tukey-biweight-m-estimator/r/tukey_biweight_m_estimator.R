# Tukey biweight M-estimator (Beaton-Tukey 1974)
# R: `MASS::rlm(psi='psi.bisquare')`,
#    `robustbase::lmrob(method='MM')` uses Tukey redescending
#    with MM-estimator start.
# Python: `statsmodels.robust.robust_linear_model.RLM(M=TukeyBiweight())`,
#         from-scratch IRLS
#
# library(MASS)
# fit <- rlm(y ~ ., data = df, psi = psi.bisquare, c = 4.685)
