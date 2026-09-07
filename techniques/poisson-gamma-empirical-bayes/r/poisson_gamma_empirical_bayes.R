# Poisson-gamma empirical Bayes (Reference Sec 24.19)
# Native R via DCluster / SpatialEpi / INLA; Python via pymc.
# Run with:  Rscript poisson_gamma_empirical_bayes.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DCluster::empbaysmooth     -- Clayton-Kaldor Poisson-gamma shrinkage\n")
  cat("  SpatialEpi::eBayes          -- empirical-Bayes SMR mapping\n")
  cat("  INLA::inla                  -- flexible full-Bayes disease mapping (BYM/BYM2)\n")
  cat("  MASS::glm.nb                 -- negative-binomial MLE\n")
  cat("Python:\n")
  cat("  pymc + Poisson-Gamma priors  -- fully Bayesian variant\n")
  cat("  statsmodels.discrete.count_model.NegativeBinomial\n")
  cat("Refs: Clayton, D. & Kaldor, J. (1987) 'Empirical Bayes estimates of age-\n")
  cat("      standardized relative risks for use in disease mapping', Biometrics\n")
  cat("      43(3): 671-681; Marshall, R.J. (1991) 'Mapping disease and mortality\n")
  cat("      rates using empirical Bayes estimators', Appl Statist 40(2).\n")
}
