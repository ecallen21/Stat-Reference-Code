# Conjoint / discrete choice experiment (Reference Sec 36.7)
# Native R via mlogit / apollo; Python xlogit + custom.
# Run with:  Rscript conjoint_choice.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlogit                            -- MNL, nested logit, mixed logit\n")
  cat("  apollo                            -- flexible choice modelling toolkit\n")
  cat("  ChoiceModelR                      -- Bayesian mixed logit (HB)\n")
  cat("  gmnl                              -- generalised MNL variants\n")
  cat("Python:\n")
  cat("  xlogit                            -- MNL / mixed logit / conditional logit\n")
  cat("  pylogit                           -- discrete choice models with mixing\n")
  cat("  biogeme                           -- research-grade choice model estimator\n")
  cat("  statsmodels.discrete.MNLogit      -- baseline MNL\n")
  cat("Refs: McFadden (1974) 'Conditional logit analysis of qualitative choice\n")
  cat("      behavior', in Frontiers in Econometrics; Train (2009) Discrete Choice\n")
  cat("      Methods with Simulation, CUP.\n")
}
