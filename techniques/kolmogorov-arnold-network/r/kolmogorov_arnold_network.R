# Kolmogorov-Arnold Network (Reference Sec 47.70)
# No established R implementation; Python has pykan / efficient-kan.
# Run with:  Rscript kolmogorov_arnold_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class KAN implementation in R -- the paper's authors ship Python\n")
  cat("  mgcv::gam / gss             -- additive-model / smoothing-spline analogue\n")
  cat("Python:\n")
  cat("  pykan                       -- authors' reference implementation\n")
  cat("  efficient-kan               -- faster batched B-spline evaluation\n")
  cat("  fastkan                     -- Chebyshev-basis KAN alternative\n")
  cat("  torch + custom              -- straightforward to reimplement\n")
  cat("  from-scratch                -- see kolmogorov_arnold_network.py\n")
  cat("Refs: Liu et al (2024) 'KAN: Kolmogorov-Arnold Networks', arXiv:2404.19756;\n")
  cat("      Kolmogorov (1957) Dokl Akad Nauk SSSR 114.\n")
}
