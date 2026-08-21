# Measurement invariance (Reference §19.x extra)
# R via lavaan / semTools.
# Run with:  Rscript measurement_invariance.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  semTools::measurementInvariance(model, data, group='grp')\n")
  cat("     -- runs configural / metric / scalar / strict comparison chain (LR + fit index deltas)\n")
  cat("  semTools::measEq.syntax(model, data, group='grp', group.equal='loadings' | 'intercepts' | ...)\n")
  cat("     -- generates syntax; fit with lavaan::cfa(...); compare with lavaan::anova()\n")
  cat("  semTools::partialInvariance()          -- freeing one parameter at a time (Byrne 1989)\n")
  cat("  MplusAutomation::createSyntax          -- if using Mplus alongside R\n")
  cat("Cheung-Rensvold criteria: |Delta CFI| <= 0.010 AND |Delta RMSEA| <= 0.015 to accept invariance.\n")
}
