# Random Fourier Features (Reference Sec 47.86)
# Native R: limited (custom + kernlab); Python via sklearn.
# Run with:  Rscript random_fourier_features.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kernlab                       -- exact kernel methods; RFF via custom code\n")
  cat("  fastRG                        -- Nystrom for kernel-adjacent sketching\n")
  cat("Python:\n")
  cat("  sklearn.kernel_approximation.RBFSampler      -- RFF for RBF kernel\n")
  cat("  sklearn.kernel_approximation.SkewedChi2Sampler / AdditiveChi2Sampler\n")
  cat("  sklearn.kernel_approximation.Nystroem        -- data-dependent RFF cousin\n")
  cat("  from-scratch                                 -- see random_fourier_features.py\n")
  cat("Refs: Rahimi & Recht (2007) NeurIPS; Le, Sarlos & Smola (2013) 'Fastfood',\n")
  cat("      ICML; Yu et al (2016) 'Orthogonal random features', NeurIPS.\n")
}
