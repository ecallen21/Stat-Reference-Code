# Beta-Binomial hierarchical (Reference Sec 47.132)
# Native R via VGAM / brms / rstanarm; Python via pymc / stan.
# Run with:  Rscript beta_binomial_hierarchical.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  VGAM::vglm(family=betabinomial)     -- ML for beta-binomial\n")
  cat("  brms::brm(family=beta_binomial)      -- full Bayes via Stan\n")
  cat("  rstanarm::stan_glmer                 -- hierarchical logistic\n")
  cat("  gamlss.dist::BB                      -- GAMLSS beta-binomial\n")
  cat("Python:\n")
  cat("  scipy.stats.betabinom                -- distribution + MLE via scipy.optimize\n")
  cat("  pymc.BetaBinomial                    -- Bayesian hierarchical\n")
  cat("  bambi / stan                         -- brms-style formula interfaces\n")
  cat("  from-scratch                         -- see beta_binomial_hierarchical.py\n")
  cat("Refs: Efron & Morris (1975) 'Data analysis using Stein's estimator', JASA;\n")
  cat("      Gelman et al (2013) 'Bayesian Data Analysis', 3rd ed, CRC, ch 5.\n")
}
