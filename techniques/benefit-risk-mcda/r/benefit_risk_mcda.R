# Benefit-risk assessment (Reference Sec 43.10)
# Native R: no single canonical package; custom + drugCombo + BRAT.
# Run with:  Rscript benefit_risk_mcda.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  drugCombo                        -- drug-combination benefit-risk\n")
  cat("  BRAT (benefit-risk action team)  -- FDA framework helpers\n")
  cat("  MCDA (custom via matrix ops)     -- weighted-sum aggregation\n")
  cat("  qualityTools                     -- decision-analysis primitives\n")
  cat("Python:\n")
  cat("  custom (numpy + scipy.optimize for weight elicitation)\n")
  cat("  sklearn                            -- adjacent multi-criteria helpers\n")
  cat("Refs: Mt-Isa et al. (2014) 'Balancing benefit and risk of medicines: a\n")
  cat("      systematic review and classification of available methodologies',\n")
  cat("      Pharmacoepi Drug Saf; PROTECT (2014) 'Benefit-risk methodology'.\n")
}
