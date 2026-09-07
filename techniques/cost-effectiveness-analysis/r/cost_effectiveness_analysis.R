# Cost-effectiveness analysis (Reference Sec 44.16)
# Native R via BCEA / heemod / dampack; Python via from-scratch.
# Run with:  Rscript cost_effectiveness_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  BCEA (Baio)             -- Bayesian cost-effectiveness analysis\n")
  cat("  heemod                   -- Markov cohort economic models\n")
  cat("  dampack                  -- decision-analytic modelling utilities\n")
  cat("  hesim                    -- discrete-event simulation health economics\n")
  cat("  survHE                    -- survival extrapolation for CEA\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see cost_effectiveness_analysis.py)\n")
  cat("  scipy + statsmodels for bootstrap CIs on ICER / NMB\n")
  cat("Refs: Drummond, M.F. et al. (2015) Methods for the Economic Evaluation\n")
  cat("      of Health Care Programmes, 4th ed., OUP; Briggs, A., Claxton, K.\n")
  cat("      & Sculpher, M. (2006) Decision Modelling for Health Economic\n")
  cat("      Evaluation, OUP.\n")
}
