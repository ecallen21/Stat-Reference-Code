# Bock nominal response model (Reference §22.x extra)
# R via mirt.
# Run with:  Rscript nominal_response_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mirt::mirt(data, model=1, itemtype='nominal')  -- Bock 1972 nominal response\n")
  cat("  mirt::itemplot(fit, item=1, type='trace')      -- category trace curves\n")
  cat("  mirt::fscores(fit, method='EAP')                -- EAP theta\n")
  cat("  ltm::gpcm / grm                                -- ordered polytomous (contrast)\n")
  cat("Python: pyIRT / girth libraries; brms with categorical() family for Bayesian NRM.\n")
}
