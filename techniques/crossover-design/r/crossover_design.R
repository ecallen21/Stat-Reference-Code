# 2x2 crossover design (Reference §18.x extra)
# R via Crossover, lme4, or geepack.
# Run with:  Rscript crossover_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Crossover::design.efficiency, Crossover::analyze2x2  -- design + analysis suite\n")
  cat("  lme4::lmer(y ~ period + treatment + (1 | subject), data)   -- mixed-model 2x2\n")
  cat("  nlme::lme(y ~ period + treatment, random = ~1 | subject)\n")
  cat("  geepack::geeglm(y ~ period + treatment, id = subject, corstr = 'exchangeable')\n")
  cat("  Higher-order crossovers (Williams designs, ABBA/BAAB): Crossover::williams()\n")
}
