# Two-stage cluster sampling (Reference §3.x extra)
# R via the survey package.
# Run with:  Rscript two_stage_cluster_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survey::svydesign(ids = ~ PSU + SSU, fpc = ~ N_PSU + N_SSU, data = df)\n")
  cat("  survey::svymean(~ y, design)         -- estimated mean with two-stage SE\n")
  cat("  survey::svytotal(~ y, design)         -- estimated total\n")
  cat("  survey::svyglm(y ~ x, design, family = binomial())  -- design-based logistic\n")
  cat("  survey::degf / survey::svycontrast   -- degrees of freedom and contrasts\n")
  cat("  ICC::ICCbare(psu, y)                  -- classical ICC estimator\n")
  cat("  For PPS at stage 1: svydesign(ids = ~PSU, probs = ~pi_psu, ...)\n")
}
