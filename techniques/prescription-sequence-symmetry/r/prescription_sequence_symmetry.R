# Prescription sequence symmetry analysis (Reference Sec 43.5, 43.13)
# Native R via custom (survival + lubridate); Python custom.
# Run with:  Rscript prescription_sequence_symmetry.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival + lubridate            -- prescription-date arithmetic\n")
  cat("  Epi::Lexis                       -- prescription timelines\n")
  cat("  heemod                           -- health-economic PSSA extensions\n")
  cat("Python:\n")
  cat("  pandas (custom)                  -- prescription date comparisons\n")
  cat("  scipy.stats.binomtest             -- symmetry test\n")
  cat("Refs: Hallas, J. (1996) 'Evidence of depression provoked by cardiovascular\n")
  cat("      medication: a prescription sequence symmetry analysis', Epidemiology;\n")
  cat("      Lai et al. (2021) 'Prescription sequence symmetry analysis (PSSA):\n")
  cat("      assessing and controlling for prescribing trends and duration biases',\n")
  cat("      Clin Epidemiol.\n")
}
