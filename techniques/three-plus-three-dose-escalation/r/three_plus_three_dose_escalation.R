# 3+3 Dose Escalation (Reference Sec 47.262)
# Native R via UBCRM (via reticulate) / dfcrm alt / custom; Python via from-scratch.
# Run with:  Rscript three_plus_three_dose_escalation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dfcrm::threeplusthree     -- simulate a 3+3 trial\n")
  cat("  dfcrm::getprior           -- prior for CRM but useful for setup\n")
  cat("  clinfun::dose.finding     -- generic dose-finding utilities\n")
  cat("  custom loop               -- 3+3 rules are ~20 lines\n")
  cat("Python:\n")
  cat("  UBCRM.three_plus_three\n")
  cat("  From-scratch (see three_plus_three_dose_escalation.py)\n")
  cat("Refs: Storer, B.E. (1989) 'Design and analysis of phase I clinical\n")
  cat("      trials', Biometrics 45(3), 925-937 (3+3 folklore predates this).\n")
}
