# SmoothGrad (Reference Sec 47.115)
# Python via captum / iNNvestigate.
# Run with:  Rscript smoothgrad_saliency.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R)                       -- build SmoothGrad manually\n")
  cat("Python:\n")
  cat("  captum.attr.NoiseTunnel + Saliency -- SmoothGrad on any attribution\n")
  cat("  iNNvestigate.analyzer.SmoothGrad   -- Keras\n")
  cat("  pytorch-grad-cam                    -- SmoothGrad wrapper\n")
  cat("  from-scratch                        -- see smoothgrad_saliency.py\n")
  cat("Refs: Smilkov, Thorat, Kim, Viegas & Wattenberg (2017) 'SmoothGrad',\n")
  cat("      arXiv:1706.03825.\n")
}
