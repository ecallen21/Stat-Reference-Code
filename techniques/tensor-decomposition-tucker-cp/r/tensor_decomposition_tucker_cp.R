# Tensor decomposition: Tucker + CP (Reference Sec 6.17)
# Native R via rTensor / multiway; Python via tensorly / from-scratch.
# Run with:  Rscript tensor_decomposition_tucker_cp.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rTensor::cp / tucker / hosvd  -- reference implementations\n")
  cat("  multiway::parafac              -- PARAFAC / CP for chemometrics\n")
  cat("  tensorregress                  -- tensor regression on responses\n")
  cat("Python:\n")
  cat("  tensorly.decomposition.parafac / tucker / non_negative_parafac\n")
  cat("  tensorly-torch                  -- GPU-backed tensor operations\n")
  cat("  scikit-tensor (older)\n")
  cat("Refs: Tucker, L.R. (1966) 'Some mathematical notes on three-mode factor\n")
  cat("      analysis', Psychometrika 31; Harshman, R.A. (1970) PARAFAC\n")
  cat("      technical report; Kolda & Bader (2009) 'Tensor decompositions and\n")
  cat("      applications', SIAM Review 51(3).\n")
}
