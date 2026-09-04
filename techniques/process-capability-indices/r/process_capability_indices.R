# Process capability indices (Reference Sec 37.7 / 37.12)
# Native R via qcc / SixSigma; Python via pyspc.
# Run with:  Rscript process_capability_indices.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qcc::process.capability      -- Cp, Cpk, Cpu, Cpl (+ visual)\n")
  cat("  SixSigma::ss.ca.cp / ss.ca.cpk -- Six Sigma-oriented\n")
  cat("Python:\n")
  cat("  pyspc                          -- Cp / Cpk helpers\n")
  cat("  matplotlib + custom            -- manual histogram + spec lines\n")
  cat("Refs: Kane, V.E. (1986) 'Process capability indices', J Qual Tech;\n")
  cat("      Taguchi, G. (1986) 'Introduction to Quality Engineering' (Cpm concept);\n")
  cat("      Montgomery, D.C. (2020) 'Introduction to SQC', 8th ed.\n")
}
