# Functional ANOVA (Reference Sec 31.4)
# Native R via fda / fda.usc; Python via scikit-fda.
# Run with:  Rscript functional_anova.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda::Fperm.fd                -- functional F-test with permutation\n")
  cat("  fda.usc::anova.RPm           -- Bonferroni-corrected pointwise ANOVA\n")
  cat("  refund::pfr                  -- Bayesian functional ANOVA (adjacent)\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- functional_data_analysis.inference F-test\n")
  cat("  fda-py                         -- pointwise F + permutation helpers\n")
  cat("Refs: Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis',\n")
  cat("      Springer, Ch. 13; Cuevas, A., Febrero, M. & Fraiman, R. (2004)\n")
  cat("      'An ANOVA test for functional data', Comp Stat Data An.\n")
}
