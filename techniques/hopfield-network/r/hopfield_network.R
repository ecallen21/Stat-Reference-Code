# Hopfield network (Reference Sec 47.90)
# Limited native R support; Python for modern-Hopfield / attention layers.
# Run with:  Rscript hopfield_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class Hopfield in mainstream R -- roll your own\n")
  cat("  neuralnet / RSNNS   -- feedforward NN utilities, no recurrent memory\n")
  cat("Python:\n")
  cat("  hflayers            -- Ramsauer modern continuous Hopfield PyTorch layer\n")
  cat("  torch + custom      -- classical binary Hopfield in a few dozen lines\n")
  cat("  from-scratch        -- see hopfield_network.py\n")
  cat("Refs: Hopfield (1982) PNAS 79(8); Amit, Gutfreund & Sompolinsky (1985)\n")
  cat("      Phys Rev A 32(2); Ramsauer et al (2021) 'Hopfield networks is all\n")
  cat("      you need', ICLR.\n")
}
