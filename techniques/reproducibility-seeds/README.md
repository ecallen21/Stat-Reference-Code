# Reproducibility: Seeds + Provenance Hashes (Reference Ch 32 MLOps)

Three things a re-run must agree on:

1. **All RNG seeds** — Python `random`, `numpy`, framework-specific
   (`torch.manual_seed`, `torch.cuda.manual_seed_all`,
   `tf.random.set_seed`, `jax.random.PRNGKey`).
2. **Deterministic ops** —
   `torch.use_deterministic_algorithms(True)`, non-deterministic CUDA
   kernels disabled, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:16:8`.
3. **Data + code + environment provenance** —
   git commit + dataset SHA-256 + lockfile (`uv.lock`, `pip freeze`).

Any one of the three missing ⇒ non-reproducible.

## Provenance manifest (compact JSON)

```
{
  "seed": 42,
  "python": "3.11.15",
  "numpy_version": "2.4.6",
  "PYTHONHASHSEED": "42",
  "git_commit": "abc123",
  "dataset_sha": "c82f...",
  "environment_lockfile_sha": "…",
  "model_weight_sha": "601d..."
}
```

Ship this alongside every trained model.

## When to use

- **Every training run**, without exception.
- **Regulatory audit** — the manifest is often a requirement.
- **Debugging drift** — a re-run with the manifest either reproduces
  the bug or isolates the non-deterministic source.

## When NOT to use

- **Never** — reproducibility is table stakes.

## Files

- `python/reproducibility_seeds.py` — from-scratch:
  - `seed_everything(seed)` — seeds Python + NumPy + `PYTHONHASHSEED`;
    comments show how to extend to Torch / CUDA.
  - `hash_bytes` / `hash_array` — SHA-256 fingerprints.
  - `reproducibility_manifest(seed)` — machine-readable capture.
  - Demo: two runs seeded `42` produce **identical model-weight SHA-256**;
    a run seeded `99` produces a different hash.
- `r/reproducibility_seeds.R` — `base::set.seed`, `RNGkind`,
  `L'Ecuyer-CMRG` for parallel workers; `digest` / `openssl` for
  hashing; `renv` for package pinning.

## Assumptions & caveats

- **Cross-hardware determinism is hard** — GPU reductions can differ
  bit-for-bit across GPU models even with all seeds set.
- **Framework-specific seeds** — remember `torch.cuda.manual_seed_all`
  for multi-GPU; `jax.random.PRNGKey` for JAX.
- **DataLoader workers** — each worker needs its own seed (torch's
  `worker_init_fn`, R's `L'Ecuyer-CMRG`).
- **PYTHONHASHSEED** must be set *before* Python starts to be
  fully deterministic; `os.environ` inside the process only
  affects subprocesses.
- **Deterministic ops slow you down** — trade-off between exact
  reproducibility and throughput; some ML shops accept "seed +
  bit-approximate" as the SLA.

## Related in this repo

- `experiment-tracking` — the tracker's params field carries the seed
  and hashes.
- `model-lineage-provenance` — the manifest fits inside a lineage node.
- `model-registry-versioning` — the manifest travels with a promoted
  model.
- `feature-store` — dataset hash pinpoints the exact feature values a
  run consumed.

## Run

```
python techniques/reproducibility-seeds/python/reproducibility_seeds.py
Rscript techniques/reproducibility-seeds/r/reproducibility_seeds.R
```

**Refs:** Buckheit, J.B. & Donoho, D.L. "WaveLab and reproducible research." Stanford tech report, 1995; Stodden, V., Guo, P. & Ma, Z. "Toward reproducible computational research." *PLOS ONE*, 2013.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
