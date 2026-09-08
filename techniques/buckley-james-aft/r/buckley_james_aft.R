# Buckley-James semi-parametric AFT (Reference Sec 11.32)
# Native R via bujar / SmoothHazard / RegularizedSCA; Python via from-scratch.
# Run with:  Rscript buckley_james_aft.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bujar                    -- boosting Buckley-James (Wang / Wang)\n")
  cat("  rms::psm                  -- includes semi-parametric fallback\n")
  cat("  survival + custom EM      -- roll-your-own BJ\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see buckley_james_aft.py)\n")
  cat("  lifelines KaplanMeierFitter for residual KM step\n")
  cat("Refs: Buckley, J. & James, I. (1979) 'Linear regression with censored\n")
  cat("      data', Biometrika 66(3): 429-436; Miller & Halpern (1982) 'Regression\n")
  cat("      with censored data', Biometrika 69(3): 521-531.\n")
}
