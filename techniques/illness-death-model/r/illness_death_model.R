# Illness-death multi-state model (Reference Sec 11.31)
# Native R via mstate / msm; Python via lifelines / from-scratch.
# Run with:  Rscript illness_death_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  msm::msm                 -- multi-state Markov / panel + covariates\n")
  cat("  mstate                    -- multi-state Cox modelling\n")
  cat("  eha                       -- multistate + parametric alternatives\n")
  cat("  survival::coxph(strata=trans) -- Cox-based multi-state fits\n")
  cat("Python:\n")
  cat("  lifelines multi-state extensions\n")
  cat("  From-scratch numpy / scipy.linalg.expm (see illness_death_model.py)\n")
  cat("  sksurv multi-state utilities (in development)\n")
  cat("Refs: Kalbfleisch & Prentice (2002) The Statistical Analysis of Failure\n")
  cat("      Time Data ch 8; Andersen & Keiding (2002) 'Multi-state models for\n")
  cat("      event history analysis', Stat Meth Med Res 11.\n")
}
