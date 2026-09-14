# Multitaper Spectral Density (Reference Sec 47.332)
# Native R via multitaper / astsa; Python via mne / nitime / spectrum / scipy.signal.
# Run with:  Rscript multitaper_spectral_density.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  multitaper (Rahim)         -- DPSS-tapered spectrum estimation\n")
  cat("  astsa::mvspec              -- multivariate multitaper + confidence bands\n")
  cat("  bspec                      -- Bayesian multitaper alternative\n")
  cat("Python:\n")
  cat("  scipy.signal.windows.dpss   -- Slepian tapers\n")
  cat("  mne.time_frequency.psd_multitaper\n")
  cat("  nitime.algorithms.spectral.multi_taper_psd\n")
  cat("  spectrum library\n")
  cat("  From-scratch (see multitaper_spectral_density.py)\n")
  cat("Refs: Thomson, D.J. (1982) 'Spectrum estimation and harmonic\n")
  cat("      analysis', Proc IEEE 70(9);  Percival, D.B. & Walden, A.T.\n")
  cat("      (1993) 'Spectral Analysis for Physical Applications', Cambridge.\n")
}
