# Probability integral transform + inverse-CDF sampling
# R: `pnorm/pexp/pweibull/...` for PIT (F(x)),
#    `qnorm/qexp/qweibull/...` for inverse-CDF (F^{-1}(u)),
#    `stats::qqplot`, `randtoolbox::sobol` (QMC + inverse-CDF pipeline).
# Python: `scipy.stats.<dist>.cdf` / `.ppf`,
#         `scipy.special.ndtr` / `ndtri`, from-scratch
