# MR-Egger regression (Reference Sec 47.64)
# Native R via MendelianRandomization / TwoSampleMR; Python is limited.
# Run with:  Rscript mr_egger_pleiotropy.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MendelianRandomization::mr_egger  -- IVW / Egger / weighted-median / MBE\n")
  cat("  TwoSampleMR::mr                   -- MRC-IEU harmonised MR pipeline\n")
  cat("  MRPRESSO                          -- outlier-pleiotropy detection + BMR\n")
  cat("  mendelianRandomization            -- SIMEX for NOME correction\n")
  cat("Python:\n")
  cat("  pymr                              -- limited MR toolkit\n")
  cat("  from-scratch                      -- see mr_egger_pleiotropy.py\n")
  cat("Refs: Bowden, Davey Smith & Burgess (2015) Int J Epidemiol 44(2);\n")
  cat("      Hemani, Bowden & Davey Smith (2018) Hum Mol Genet 27(R2).\n")
}
