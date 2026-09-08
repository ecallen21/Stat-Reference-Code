# Structural Equation Modeling (Reference Sec 47.137).
#
# Joreskog 1970 'A general method for analysis of covariance
# structures'. Measurement + structural models fit by ML.
#
# Uses `lavaan` (recommended) or `sem` for maximum-likelihood SEM.

if (!requireNamespace("lavaan", quietly = TRUE)) install.packages("lavaan")
library(lavaan)                                       # ML-SEM engine

set.seed(0)
n <- 400
xi <- rnorm(n)
lam_true <- c(0.9, 0.8, 0.7, 0.6, 0.5)
theta_true <- c(0.19, 0.36, 0.51, 0.64, 0.75)
X <- outer(xi, lam_true) + matrix(rnorm(n * 5, sd = sqrt(theta_true)), n, 5, byrow = TRUE)
colnames(X) <- paste0("x", 1:5)
dat <- as.data.frame(X)

# One-factor CFA
mod <- '
    F =~ x1 + x2 + x3 + x4 + x5
'
fit <- cfa(mod, data = dat, std.lv = TRUE)
cat("=== One-factor CFA via lavaan (Joreskog 1970) ===\n")
summary(fit, fit.measures = TRUE, standardized = TRUE)
