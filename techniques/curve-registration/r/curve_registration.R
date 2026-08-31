# Curve registration (Reference Sec 31.11)
# Native R via fda::landmarkreg / register.fd; Python via fdasrsf.
# Run with:  Rscript curve_registration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda::landmarkreg              -- landmark-based piecewise-linear warp\n")
  cat("  fda::register.fd              -- continuous registration via minimum-eigenvalue\n")
  cat("  fdasrvf (R port)              -- Srivastava-Klassen SRVF phase-amplitude decomp\n")
  cat("Python:\n")
  cat("  fdasrsf                        -- SRSF elastic curve alignment (Kurtek)\n")
  cat("  scikit-fda                    -- landmark / elastic warping registration\n")
  cat("Refs: Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis',\n")
  cat("      Springer, Ch. 7; Srivastava, A. & Klassen, E. (2016) 'Functional and\n")
  cat("      Shape Data Analysis', Springer (SRSF / SRVF framework).\n")
}
