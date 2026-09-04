# Information bottleneck (Reference Sec 34.12)
# Native R limited; Python via information-bottleneck / deep IB.
# Run with:  Rscript information_bottleneck.R

if (sys.nframe() == 0) {
  cat("R packages: no dedicated CRAN implementation; use reticulate + Python.\n")
  cat("  infotheo                     -- discrete MI building blocks\n")
  cat("Python:\n")
  cat("  information-bottleneck        -- Bialek-lab discrete IB\n")
  cat("  deep-info-bottleneck          -- Alemi 2016 VIB in TF/PyTorch\n")
  cat("  IDTxl                          -- info-dynamics toolkit\n")
  cat("Refs: Tishby, N., Pereira, F.C. & Bialek, W. (1999) 'The information bottleneck\n")
  cat("      method', 37th Allerton Conf; Alemi, A. et al. (2017) 'Deep variational\n")
  cat("      information bottleneck', ICLR.\n")
}
