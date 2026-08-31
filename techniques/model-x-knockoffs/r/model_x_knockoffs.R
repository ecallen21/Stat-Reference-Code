# Model-X knockoffs (Reference Sec 32.5)
# Native R via knockoff; Python via knockpy.
# Run with:  Rscript model_x_knockoffs.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  knockoff                     -- Candes-Fan-Janson-Lv reference\n")
  cat("  KOBT                          -- knockoff Boosting Trees\n")
  cat("Python:\n")
  cat("  knockpy                       -- Model-X + Gaussian + second-order knockoffs\n")
  cat("  celer / hdlasso                -- fast LASSO for the importance step\n")
  cat("Refs: Candes, E.J., Fan, Y., Janson, L. & Lv, J. (2018) 'Panning for gold:\n")
  cat("      Model-X knockoffs for high-dimensional controlled variable selection', JRSS-B;\n")
  cat("      Barber, R.F. & Candes, E.J. (2015) 'Controlling the FDR via knockoffs',\n")
  cat("      Annals of Statistics.\n")
}
