# Imitation learning: BC + DAgger + inverse RL notes (Reference §28.8)
# R via reticulate + Python.
# Run with:  Rscript imitation_learning.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R package for imitation learning; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  imitation (Wang / HumanCompatibleAI)  -- BC, DAgger, GAIL, AIRL, DensityIRL\n")
  cat("  stable-baselines3-contrib             -- SAC-BC, DQN-BC, GAIL wrappers\n")
  cat("  d3rlpy                                 -- offline / imitation with a rich algo suite\n")
  cat("Foundational refs:\n")
  cat("  Pomerleau 1989 ALVINN (BC for autonomous driving)\n")
  cat("  Ross-Gordon-Bagnell 2011 DAgger\n")
  cat("  Ho-Ermon 2016 GAIL (generative adversarial imitation)\n")
  cat("  Ziebart 2008 MaxEnt inverse RL\n")
  cat("Applications: autonomous driving (Waymo, Wayve), robot manipulation demos,\n")
  cat("              behavioural priors for RL warm-start, LLM SFT from human demonstrations.\n")
}
