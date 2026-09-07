# Principal stratification -- CACE / LATE (Reference Sec 15.32)
# Native R via ivreg / AER / eStrata; Python via linearmodels.IV2SLS.
# Run with:  Rscript principal_stratification_cace.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ivreg::ivreg / AER::ivreg     -- Wald / 2SLS for CACE\n")
  cat("  eStrata                        -- Bayesian principal strata\n")
  cat("  noncomplyR                    -- likelihood CACE with covariates\n")
  cat("Python:\n")
  cat("  linearmodels.iv.IV2SLS         -- classical 2SLS\n")
  cat("  statsmodels.sandbox.regression.gmm -- GMM IV\n")
  cat("Refs: Frangakis, C.E. & Rubin, D.B. (2002) 'Principal stratification\n")
  cat("      in causal inference', Biometrics 58(1); Angrist, Imbens & Rubin\n")
  cat("      (1996) 'Identification of causal effects using instrumental\n")
  cat("      variables', JASA 91(434): 444-455.\n")
}
