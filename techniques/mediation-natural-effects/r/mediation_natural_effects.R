# Mediation -- natural direct/indirect effects (Reference Sec 15.38)
# Native R via mediation / medflex / regmedint; Python via DoWhy.
# Run with:  Rscript mediation_natural_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mediation::mediate            -- Imai, Keele & Tingley (nonparametric)\n")
  cat("  medflex::neImpute / neWeight  -- imputation and weighting NDE/NIE\n")
  cat("  regmedint::regmedint          -- VanderWeele closed-form (interactions)\n")
  cat("  paths                          -- multiple-mediator path-specific effects\n")
  cat("Python:\n")
  cat("  DoWhy (Microsoft) -- graphical mediation utilities\n")
  cat("  causallib.estimation.overlap_weights + custom NDE/NIE\n")
  cat("Refs: VanderWeele, T.J. (2015) Explanation in Causal Inference: Methods for\n")
  cat("      Mediation and Interaction, OUP; Imai, Keele & Tingley (2010)\n")
  cat("      'A general approach to causal mediation analysis', Psych Meth 15(4).\n")
}
