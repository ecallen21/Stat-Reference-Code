# Information geometry (Reference Sec 34.15)
# Native R limited; Python via jax + custom.
# Run with:  Rscript information_geometry.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  geometry                     -- computational geometry helpers\n")
  cat("  glmnet, brms                 -- exponential-family GLMs sharing Fisher-info machinery\n")
  cat("Python:\n")
  cat("  geomstats                     -- Riemannian geometry incl. Fisher-Rao metric\n")
  cat("  jax.example_libraries.optimizers.natural_gradient (community)\n")
  cat("  pytorch/kfac                  -- Kronecker-factored approximate curvature (adjacent)\n")
  cat("Refs: Amari, S.-I. & Nagaoka, H. (2007) 'Methods of Information Geometry', AMS;\n")
  cat("      Amari, S.-I. (2016) 'Information Geometry and Its Applications', Springer.\n")
}
