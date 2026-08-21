# Split-plot ANOVA (Reference §18.x extra)
# R via afex, lmerTest, or aov with Error().
# Run with:  Rscript split_plot_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  aov(y ~ A * B + Error(WP / B))                     -- classical two-error ANOVA\n")
  cat("  lme4::lmer(y ~ A * B + (1 | WP)) + lmerTest::anova -- mixed-model split-plot\n")
  cat("  afex::aov_car(y ~ A * B + Error(WP / B), data)     -- ez interface\n")
  cat("  agricolae::sp.plot(block, wp_factor, sp_factor, response)\n")
  cat("Python: statsmodels.stats.anova_lm with mixed formulas; pymer4 for lmer bindings.\n")
}
