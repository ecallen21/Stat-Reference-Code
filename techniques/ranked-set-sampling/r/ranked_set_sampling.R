# Ranked-set sampling (Reference Sec 27.9)
# Native R via RSSampling; Python via from-scratch.
# Run with:  Rscript ranked_set_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RSSampling                 -- classical + median RSS + estimators\n")
  cat("  rssampling                 -- lighter interface for RSS studies\n")
  cat("  sampling                    -- includes generalised RSS designs\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see ranked_set_sampling.py)\n")
  cat("  No mature RSS-dedicated PyPI package as of 2024\n")
  cat("Refs: McIntyre, G.A. (1952) 'A method of unbiased selective sampling\n")
  cat("      using ranked sets', Aust J Agric Res 3(4): 385-390; Takahasi, K.\n")
  cat("      & Wakimoto, K. (1968) 'On unbiased estimates of the population\n")
  cat("      mean based on the sample stratified by means of ordering', Ann\n")
  cat("      Inst Stat Math 20; Chen, Bai & Sinha (2004) Ranked Set Sampling.\n")
}
