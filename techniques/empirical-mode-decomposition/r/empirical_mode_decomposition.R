# Empirical Mode Decomposition (Reference Sec 47.93)
# Native R via Rlibeemd / EMD; Python via PyEMD.
# Run with:  Rscript empirical_mode_decomposition.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Rlibeemd::emd / eemd     -- Fast C++ EMD / Ensemble EMD\n")
  cat("  EMD                      -- classical implementation + Hilbert spectrum\n")
  cat("  hht                      -- Hilbert-Huang transform pipeline\n")
  cat("Python:\n")
  cat("  PyEMD / EMD-signal       -- EMD, EEMD, CEEMDAN, MVEMD\n")
  cat("  pyhht                    -- Hilbert-Huang transform\n")
  cat("  from-scratch             -- see empirical_mode_decomposition.py\n")
  cat("Refs: Huang et al (1998) Proc R Soc A 454(1971); Wu & Huang (2009)\n")
  cat("      'Ensemble EMD', Adv Adapt Data Anal 1(1).\n")
}
