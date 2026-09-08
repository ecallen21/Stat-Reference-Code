# Welch PSD (Reference Sec 47.94)
# Native R via stats / spectral / psd; Python via scipy.
# Run with:  Rscript welch_power_spectral_density.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::spectrum(x, spans=...)  -- smoothed periodogram (Daniell windows)\n")
  cat("  spectral                       -- Welch, multitaper, adaptive PSD\n")
  cat("  psd::pspectrum                 -- adaptive Sine-multitaper PSD\n")
  cat("  signal (R port)                -- Welch, pwelch-alike\n")
  cat("Python:\n")
  cat("  scipy.signal.welch             -- reference implementation\n")
  cat("  scipy.signal.periodogram       -- single-shot periodogram\n")
  cat("  mne.time_frequency.psd_welch   -- EEG/MEG-oriented Welch\n")
  cat("  from-scratch                   -- see welch_power_spectral_density.py\n")
  cat("Refs: Welch (1967) IEEE Trans Audio Electroacoust 15(2); Percival & Walden\n")
  cat("      (1993) 'Spectral Analysis for Physical Applications', Cambridge.\n")
}
