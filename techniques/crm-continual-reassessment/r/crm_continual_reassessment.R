# Continual Reassessment Method (Reference Sec 47.261)
# Native R via dfcrm / bcrm / trialr; Python via UBCRM / from-scratch.
# Run with:  Rscript crm_continual_reassessment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dfcrm::crm              -- classical CRM, ESS calibration\n")
  cat("  bcrm                    -- Bayesian CRM with two-parameter model\n")
  cat("  trialr::stan_crm        -- Stan-backed CRM with full posterior\n")
  cat("  crmPack                 -- object-oriented, extensible framework\n")
  cat("Python:\n")
  cat("  UBCRM                    -- pip package for CRM simulation\n")
  cat("  From-scratch (see crm_continual_reassessment.py)\n")
  cat("Refs: O'Quigley, J., Pepe, M. & Fisher, L. (1990) 'Continual\n")
  cat("      reassessment method: A practical design for phase 1 clinical\n")
  cat("      trials in cancer', Biometrics 46(1), 33-48.\n")
}
