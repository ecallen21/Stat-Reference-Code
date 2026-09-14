# Hilbert Transform / Analytic Signal (Reference Sec 47.331)
# Native R via seewave / signal; Python via scipy.signal / from-scratch.
# Run with:  Rscript hilbert_transform_analytic.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  signal::hilbert           -- analytic signal via FFT\n")
  cat("  seewave::env              -- envelope via Hilbert transform\n")
  cat("  hht (Hilbert-Huang)       -- with EMD for non-stationary\n")
  cat("Python:\n")
  cat("  scipy.signal.hilbert\n")
  cat("  emd-signal / PyEMD.hht\n")
  cat("  From-scratch FFT (see hilbert_transform_analytic.py)\n")
  cat("Refs: Hilbert, D. (1912) 'Grundzuge einer allgemeinen Theorie der\n")
  cat("      linearen Integralgleichungen';  Gabor, D. (1946) 'Theory of\n")
  cat("      communication', J IEE 93.\n")
}
