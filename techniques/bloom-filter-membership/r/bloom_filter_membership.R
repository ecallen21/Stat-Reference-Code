# Bloom filter (Reference Sec 47.72)
# Native R via bloomfilter; Python via pybloom / from-scratch.
# Run with:  Rscript bloom_filter_membership.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bloomfilter               -- pure-R Bloom filter\n")
  cat("  BLOOM (RcppBloom)         -- C++ speed for large streams\n")
  cat("Python:\n")
  cat("  pybloom / pybloom_live    -- scalable Bloom + counting Bloom\n")
  cat("  pybloomfiltermmap3        -- mmap-backed for shared use across processes\n")
  cat("  cuckoofilter              -- deletion + slightly less memory\n")
  cat("  from-scratch              -- see bloom_filter_membership.py\n")
  cat("Refs: Bloom (1970) CACM 13(7); Kirsch & Mitzenmacher (2006) 'Less hashing,\n")
  cat("      same performance: Building a better Bloom filter', ESA.\n")
}
