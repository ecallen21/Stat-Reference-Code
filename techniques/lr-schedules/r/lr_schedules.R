# Learning-rate schedules (Reference §27.x extra)
# R via torch or keras3.
# Run with:  Rscript lr_schedules.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::lr_scheduler_step_lr(opt, step_size, gamma)\n")
  cat("  torch::lr_scheduler_cosine_annealing_lr(opt, T_max, eta_min)\n")
  cat("  torch::lr_scheduler_one_cycle_lr(opt, max_lr, total_steps)\n")
  cat("  torch::lr_scheduler_reduce_on_plateau(opt, patience, factor)\n")
  cat("  keras3::callback_learning_rate_scheduler(schedule=fn)\n")
  cat("Python:\n")
  cat("  torch.optim.lr_scheduler.{StepLR, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts,\n")
  cat("                             LinearLR, LambdaLR, ReduceLROnPlateau, MultiStepLR}\n")
  cat("  huggingface transformers.get_scheduler / get_cosine_schedule_with_warmup / get_polynomial_...\n")
  cat("  optax (JAX): cosine_decay_schedule, warmup_cosine_decay_schedule, exponential_decay.\n")
}
