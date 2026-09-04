# Heterogeneous treatment effect / uplift (Reference Sec 44.7)
# Native R via grf / uplift; Python causalml + econml + custom.
# Run with:  Rscript hte_uplift.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  grf::causal_forest                -- Athey-Wager honest causal forest\n")
  cat("  uplift                             -- uplift models + Qini curves\n")
  cat("  causalToolbox                      -- T/S/X-learners on any base learner\n")
  cat("Python:\n")
  cat("  causalml (UpliftRandomForestClassifier)\n")
  cat("  econml (CausalForestDML, MetaLearners)\n")
  cat("  sklearn + custom meta-learners\n")
  cat("Refs: Athey & Imbens (2016) 'Recursive partitioning for heterogeneous causal\n")
  cat("      effects', PNAS; Kunzel, Sekhon, Bickel & Yu (2019) 'Metalearners for\n")
  cat("      estimating heterogeneous treatment effects using machine learning', PNAS.\n")
}
