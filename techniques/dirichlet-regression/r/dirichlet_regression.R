# Dirichlet Regression (Reference Sec 47.324)
# Native R via DirichletReg / brms; Python via dirichlet / PyMC / from-scratch.
# Run with:  Rscript dirichlet_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DirichletReg                    -- MLE Dirichlet regression\n")
  cat("  brms (family = dirichlet)       -- Bayesian Dirichlet regression via Stan\n")
  cat("  bayesplot / rstanarm            -- Bayesian visualisation and fitting\n")
  cat("Python:\n")
  cat("  dirichlet (PyPI)                -- Dirichlet distribution MLE\n")
  cat("  PyMC / NumPyro                  -- Bayesian Dirichlet regression\n")
  cat("  From-scratch (see dirichlet_regression.py)\n")
  cat("Refs: Campbell, G. & Mosimann, J.E. (1987) 'Multivariate methods for\n")
  cat("      proportional shape', ASA Proc Sec Statistical Graphics;\n")
  cat("      Maier, M.J. (2014) 'DirichletReg: Dirichlet regression for\n")
  cat("      compositional data in R', J Stat Softw.\n")
}
