# Fractional polynomials (Reference Sec 5.14)
# Native R via mfp; Python via from-scratch (custom).
# Run with:  Rscript fractional_polynomials.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mfp::mfp                    -- multivariable FP for lm/glm/coxph\n")
  cat("  mfp::mfp2                   -- extension w/ interactions and priors\n")
  cat("  mfpa                        -- Bayesian FP\n")
  cat("Python:\n")
  cat("  No mature FP library; use scipy + brute-force power search or\n")
  cat("  patsy-style basis + statsmodels OLS/GLM.\n")
  cat("Refs: Royston, P. & Altman, D.G. (1994) 'Regression using fractional\n")
  cat("      polynomials of continuous covariates: parsimonious parametric\n")
  cat("      modelling', JRSS-C 43(3): 429-467; Royston & Sauerbrei (2008)\n")
  cat("      Multivariable Model-building with Fractional Polynomials, Wiley.\n")
}
