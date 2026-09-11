# Basket Trial Design (Reference Sec 47.264)
# Native R via bhmbasket / basket / rstanarm; Python via PyMC / from-scratch.
# Run with:  Rscript basket_trial_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bhmbasket                -- Bayesian hierarchical basket + EXNEX + calibrated\n")
  cat("  basket                   -- Berry / CBHM / MEM implementations\n")
  cat("  rstanarm::stan_glmer     -- generic hierarchical Bayes\n")
  cat("  brms                     -- Stan-backed hierarchical bernoulli\n")
  cat("Python:\n")
  cat("  PyMC (custom hierarchical model)\n")
  cat("  bayesian_testing (basic Bayesian A/B, not baskets)\n")
  cat("  From-scratch MH (see basket_trial_design.py)\n")
  cat("Refs: Berry, S.M. et al (2013) 'Bayesian hierarchical modeling of\n")
  cat("      patient subpopulations: Efficient designs of phase II oncology\n")
  cat("      clinical trials', Clinical Trials 10(5), 720-734; Simon, R. et al\n")
  cat("      (2016) 'The Bayesian basket design for genomic variant-driven\n")
  cat("      phase II trials', Semin Oncol 43(1), 13-18.\n")
}
