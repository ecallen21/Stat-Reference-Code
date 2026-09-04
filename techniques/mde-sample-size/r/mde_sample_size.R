# A/B sample size + MDE (Reference Sec 44.2)
# Native R via pwr; Python statsmodels + custom.
# Run with:  Rscript mde_sample_size.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pwr::pwr.t.test / pwr.2p.test / pwr.p.test -- power/n calculators\n")
  cat("  WebPower                                     -- larger power library\n")
  cat("  stats::power.t.test / power.prop.test        -- base-R equivalents\n")
  cat("Python:\n")
  cat("  statsmodels.stats.power (TTestIndPower, NormalIndPower)\n")
  cat("  scipy.stats + custom closed forms\n")
  cat("Refs: Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments,\n")
  cat("      CUP, Ch 14-15; van Belle (2008) Statistical Rules of Thumb, 2nd ed., Wiley.\n")
}
