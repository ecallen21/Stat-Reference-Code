# Pearl front-door adjustment (Reference §15.x extra)
# R via dagitty (identification) + manual estimation, or dowhy from Python.
# Run with:  Rscript front_door_criterion.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dagitty::adjustmentSets(g, exposure, outcome, type='minimal')  -- back-door / front-door sets\n")
  cat("  dagitty::isAdjustmentSet(g, Z, exposure, outcome)\n")
  cat("  causaleffect::causal.effect(y, x, G=g)                          -- symbolic do-calculus identification\n")
  cat("  bnlearn::query()                                                -- MLE + intervention on a BN\n")
  cat("Python: DoWhy (from-graph identify + estimate); pyAgrum.\n")
}
