# Network meta-analysis (Reference Sec 22.15)
# Native R via netmeta / gemtc / rjags; Python custom.
# Run with:  Rscript network_meta_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  netmeta::netmeta                  -- frequentist contrast-based NMA\n")
  cat("  gemtc                              -- Bayesian NMA via BUGS / JAGS\n")
  cat("  BUGSnet                             -- BUGS-based NMA (BayesianTools)\n")
  cat("  pcnetmeta                           -- component NMA for complex interventions\n")
  cat("Python:\n")
  cat("  custom (numpy + scipy)             -- contrast-based RE-NMA\n")
  cat("  pymare                              -- adjacent meta-analysis toolkit\n")
  cat("Refs: Lumley (2002) 'Network meta-analysis for indirect treatment comparisons',\n")
  cat("      Stat Med; Salanti (2012) 'Indirect and mixed-treatment comparison',\n")
  cat("      Research Synthesis Methods.\n")
}
