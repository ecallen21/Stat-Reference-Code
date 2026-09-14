# Thin-Plate Splines (Reference Sec 47.328)
# Native R via fields / mgcv; Python via scipy.interpolate / from-scratch.
# Run with:  Rscript thin_plate_splines.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fields::Tps                        -- classical thin-plate spline\n")
  cat("  mgcv::gam(..., bs='tp')           -- penalised TPS via GAM framework\n")
  cat("  spatstat / geoR                   -- kriging-flavoured alternatives\n")
  cat("Python:\n")
  cat("  scipy.interpolate.RBFInterpolator(kernel='thin_plate_spline')\n")
  cat("  scipy.interpolate.Rbf(function='thin_plate')     -- legacy interface\n")
  cat("  From-scratch (see thin_plate_splines.py)\n")
  cat("Refs: Duchon, J. (1977) 'Splines minimizing rotation-invariant\n")
  cat("      semi-norms in Sobolev spaces', in Constructive Theory of\n")
  cat("      Functions of Several Variables;  Wahba, G. (1990) 'Spline Models\n")
  cat("      for Observational Data', SIAM.\n")
}
