# Guardrail monitoring for A/B tests (Reference Sec 44.15)
# Native R via stats + qcc; Python custom + scipy.
# Run with:  Rscript guardrail_monitoring.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::binom.test / prop.test    -- one-sample proportion CIs\n")
  cat("  qcc                                -- sequential SPC-style monitoring\n")
  cat("  Hmisc::binconf                     -- Wilson CI convenience\n")
  cat("Python:\n")
  cat("  scipy.stats.beta.ppf              -- Clopper-Pearson via Beta\n")
  cat("  statsmodels.stats.proportion       -- Wilson / Jeffreys CIs\n")
  cat("  eppo-sdk / statsig / growthbook    -- guardrail dashboards\n")
  cat("Refs: Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments, CUP;\n")
  cat("      Fabijan, Dmitriev, Olsson & Bosch (2017) 'The evolution of continuous\n")
  cat("      experimentation in software product development', ICSE.\n")
}
