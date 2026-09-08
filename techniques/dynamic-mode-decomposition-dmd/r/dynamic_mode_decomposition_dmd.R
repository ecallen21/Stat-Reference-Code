# DMD (Reference Sec 47.126)
# Python via pydmd; R limited.
# Run with:  Rscript dynamic_mode_decomposition_dmd.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no established DMD in mainstream R -- roll your own with base SVD\n")
  cat("Python:\n")
  cat("  pydmd                 -- Kutz group reference (DMDc, mrDMD, EDMD, HODMD)\n")
  cat("  scipy.linalg.svd      -- for the truncated-SVD step\n")
  cat("  torchdyn              -- GPU-friendly DMD variants\n")
  cat("  from-scratch          -- see dynamic_mode_decomposition_dmd.py\n")
  cat("Refs: Schmid (2010) J Fluid Mech 656; Tu et al (2014) J Comp Dyn 1(2).\n")
}
