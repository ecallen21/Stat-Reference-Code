# Information criteria (Reference Sec 34.5)
# Native R has AIC, BIC, AICc, loo::waic, DIC via rjags/rstan.
# Run with:  Rscript information_criteria.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::AIC / stats::BIC       -- built-in\n")
  cat("  MuMIn::AICc                    -- small-n corrected AIC\n")
  cat("  loo::waic / loo::loo           -- Watanabe WAIC + PSIS-LOO\n")
  cat("  rjags / rstan                  -- DIC via posterior sampling\n")
  cat("Python:\n")
  cat("  statsmodels OLS().fit().aic / .bic\n")
  cat("  pymc.waic / pymc.loo\n")
  cat("  arviz.waic / arviz.loo\n")
  cat("Refs: Akaike, H. (1974) 'A new look at the statistical model identification', IEEE TAC;\n")
  cat("      Schwarz, G. (1978) 'Estimating the dimension of a model', Ann Stat;\n")
  cat("      Watanabe, S. (2010) 'Asymptotic equivalence of Bayes cross validation\n")
  cat("      and widely applicable information criterion', JMLR.\n")
}
