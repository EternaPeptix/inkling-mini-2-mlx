# Rules for swarm implementers

* Read docs/OBJECTIVES.md. Never run Python, builds or benchmarks on this Mac. Use `LANE=<your lane> scripts/r [--gpu] <cmd>`
  from your worktree root, run directly as a zsh script (never `bash scripts/r`). Use `--gpu` for anything touching Metal.
  The lock serializes the one GPU between lanes: keep GPU commands short (minutes), never leave a process running.
* Only touch 512S1 inside `~/loopling-opt/src-<lane>` and `~/loopling-opt/runs/<lane>/`. Read-only: `~/loopling/models`.
  Never touch `~/inkling-lab`, the process on port 8080, `~/exo`, `~/.exo`, or any other host.
* Change runtime code only in `vendor/mlx_lm` (and new files in `bench/` or your own module). Every change behind an
  environment flag that defaults OFF until it passes the exact gate; say which flag.
* Before claiming a speedup: `bench/bench.py` with and without your flag in the same session, >= 3 runs each, plus
  `bench/exact.py` against `runs/baseline.json`. Report medians, min/max, and the exact-gate result.
* Commit in your worktree, messages ending with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
* If a command is denied or blocked, do not work around it; report it.
