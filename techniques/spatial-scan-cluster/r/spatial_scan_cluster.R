# Kulldorff spatial scan statistic (Reference §23.13)
# R via SpatialEpi::kulldorff, DCluster, or the standalone SaTScan program.
# Run with:  Rscript spatial_scan_cluster.R

if (sys.nframe() == 0) {
  cat("R packages / tools:\n")
  cat("  SpatialEpi::kulldorff(geo, cases, pop, expected.cases, pop.upper.bound, ...)\n")
  cat("  DCluster::opgam / DCluster::besagnewell            -- Besag-Newell cluster tests\n")
  cat("  smerc::scan.test                                   -- SaTScan-like scan in R\n")
  cat("  SaTScan (standalone, https://www.satscan.org)      -- reference implementation\n")
}
