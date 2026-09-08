# Elo / Glicko rating systems (Reference Sec 47.66)
# Native R via PlayerRatings; Python via skelo / trueskill / openskill.
# Run with:  Rscript elo_glicko_rating.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PlayerRatings::runGlicko / runGL  -- Elo, Glicko, Stephenson\n")
  cat("  elo::elo.run                      -- Elo pipeline with covariates\n")
  cat("  EloRating::elo.seq                -- animal-behaviour dominance ratings\n")
  cat("Python:\n")
  cat("  skelo             -- Glicko + Glicko-2 pipeline\n")
  cat("  trueskill         -- Xbox Live TrueSkill (team + free-for-all)\n")
  cat("  openskill         -- Plackett-Luce Bayesian rating suite\n")
  cat("  from-scratch      -- see elo_glicko_rating.py\n")
  cat("Refs: Elo (1978) 'The Rating of Chessplayers, Past and Present', Arco;\n")
  cat("      Glickman (1999) Appl Stat 48(3); Herbrich et al (2007) 'TrueSkill'\n")
  cat("      NeurIPS.\n")
}
