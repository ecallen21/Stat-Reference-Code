# GWAS (Reference Sec 40.1, 40.23)
# PLINK (external) is canonical; qqman + SNPassoc for R workflows.
# Run with:  Rscript gwas.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  qqman::manhattan, qq            -- Manhattan + QQ plots\n")
  cat("  GWASTools, SNPassoc, GENESIS    -- mixed-model GWAS with kinship\n")
  cat("  bigsnpr::snp_manhattan, snp_qq  -- large-scale GWAS toolkit\n")
  cat("Python:\n")
  cat("  hail::linear_regression_rows    -- distributed GWAS\n")
  cat("  pandas-plink, pysnptools        -- PLINK bed/bim/fam I/O\n")
  cat("  custom scipy                    -- per-SNP regression scan\n")
  cat("External:\n")
  cat("  PLINK 2.0, REGENIE, SAIGE       -- production GWAS pipelines\n")
  cat("Refs: Uffelmann et al. (2021) 'Genome-wide association studies', Nat Rev Methods\n")
  cat("      Primers; Price et al. (2006) PCA correction for stratification, Nat Genet.\n")
}
