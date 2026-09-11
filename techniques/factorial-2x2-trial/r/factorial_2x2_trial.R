# 2x2 Factorial Trial (Reference Sec 47.267)
# Native R via base lm + anova / afex / emmeans; Python via statsmodels.
# Run with:  Rscript factorial_2x2_trial.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  base::lm + anova         -- OLS fit + Type I/II/III sums of squares\n")
  cat("  afex::aov_ez             -- ANOVA helpers with interaction contrasts\n")
  cat("  emmeans::emmeans         -- estimated marginal means + contrasts\n")
  cat("  pwr::pwr.f2.test         -- F-test power for factorial designs\n")
  cat("Python:\n")
  cat("  statsmodels.formula.api.ols('y ~ A*B', data)\n")
  cat("  statsmodels.stats.anova.anova_lm\n")
  cat("  From-scratch OLS (see factorial_2x2_trial.py)\n")
  cat("Refs: Piantadosi, S. (2005) 'Clinical Trials: A Methodologic Perspective',\n")
  cat("      2nd ed, Wiley; McAlister, F.A. et al (2003) 'Analysis and reporting\n")
  cat("      of factorial trials: A systematic review', JAMA 289(19), 2545-2553.\n")
}
