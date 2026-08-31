# Epistemic vs Aleatoric uncertainty decomposition (Reference Ch 29 UQ)
# R via reticulate + Python; native R support is limited.
# Run with:  Rscript epistemic_aleatoric.R

if (sys.nframe() == 0) {
  cat("R packages: limited native; the decomposition itself is a few lines of code.\n")
  cat("  torch (R port) + custom loop -- MC / ensemble variance decomposition\n")
  cat("Python:\n")
  cat("  uncertainty-toolbox         -- calibration + sharpness for aleatoric/epistemic\n")
  cat("  laplace-torch               -- posterior-based decomposition on the last layer\n")
  cat("  pyro                        -- SVI-based BNN with variance decomposition helper\n")
  cat("  bayesian-torch              -- BNN layers with MC posterior sampling\n")
  cat("Refs: Kendall, A. & Gal, Y. (2017) 'What Uncertainties Do We Need in Bayesian\n")
  cat("      Deep Learning for Computer Vision?', NeurIPS;\n")
  cat("      Depeweg, S. et al. (2018) 'Decomposition of Uncertainty in Bayesian Deep\n")
  cat("      Learning for Efficient and Risk-sensitive Learning', ICML;\n")
  cat("      Houlsby, N. et al. (2011) 'Bayesian Active Learning for Classification\n")
  cat("      and Preference Learning (BALD)'.\n")
}
