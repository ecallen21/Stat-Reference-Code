# MMD two-sample test (Reference Sec 47.65)
# Native R via kernlab; Python via hyppo / torch-two-sample.
# Run with:  Rscript mmd_two_sample_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kernlab::kmmd       -- Gretton MMD^2 with bootstrap p\n")
  cat("  eummd               -- exact unbiased MMD test\n")
  cat("  Ball::bd.test       -- ball divergence two-sample test cousin\n")
  cat("Python:\n")
  cat("  hyppo.ksample.MMD   -- MMD^2 with permutation p\n")
  cat("  torch-two-sample    -- MMD, energy, classifier tests\n")
  cat("  falkonhep           -- linear-time MMD block estimator\n")
  cat("  from-scratch        -- see mmd_two_sample_test.py\n")
  cat("Refs: Gretton, Borgwardt, Rasch, Scholkopf & Smola (2012) JMLR 13;\n")
  cat("      Chwialkowski et al (2015) NeurIPS.\n")
}
