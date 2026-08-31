# Reproducibility: seeds + provenance hashes (Reference Ch 32 MLOps)
# Native R is quite capable; Python via seed helpers + hashing libraries.
# Run with:  Rscript reproducibility_seeds.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  base::set.seed              -- global RNG seed\n")
  cat("  RNGkind                     -- pick RNG family (default vs L'Ecuyer)\n")
  cat("  future.apply / parallel     -- seed each worker (RNGkind='L'Ecuyer-CMRG')\n")
  cat("  digest, openssl             -- SHA-256 / MD5 fingerprints of data + model\n")
  cat("  renv                         -- pin package versions\n")
  cat("Python:\n")
  cat("  pytorch-lightning.seed_everything    -- seeds python + numpy + torch + cuda\n")
  cat("  transformers.set_seed                 -- HF unified seeding\n")
  cat("  dvc                                   -- pin data + intermediate artifact hashes\n")
  cat("  reprozip                              -- capture full environment for replay\n")
  cat("Refs: Buckheit, J. & Donoho, D. (1995) 'WaveLab and Reproducible Research';\n")
  cat("      Stodden, V., Guo, P. & Ma, Z. (2013) 'Toward Reproducible Computational Research'.\n")
}
