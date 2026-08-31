# Functional basis smoothing (Reference Sec 31.1)
# Native R via fda / mgcv; Python via scikit-fda.
# Run with:  Rscript functional_basis_smoothing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda::smooth.basis            -- Ramsay-Silverman B-spline / Fourier smoothing\n")
  cat("  fda::fdPar + Data2fd          -- roughness-penalty control\n")
  cat("  mgcv::gam(s(t, bs='ps'))     -- Wood P-spline smoother\n")
  cat("  splines::bs                    -- base R B-spline basis\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- BSplineBasis + FDataBasis + smoothing helpers\n")
  cat("  patsy dmatrix('bs(t, df=k)') -- design-matrix B-spline basis\n")
  cat("Refs: Eilers, P.H.C. & Marx, B.D. (1996) 'Flexible smoothing with B-splines\n")
  cat("      and penalties', Statistical Science;\n")
  cat("      Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis', Ch. 4-5.\n")
}
