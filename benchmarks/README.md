# Benchmarks

Reproduction harnesses for the benchmarks called out in the proposal:

- `agentdojo_flow.py`. AgentDojo on top of `aisafepy.flow`. Target:
  ASR ≤ 5% with ≥ 90% utility against a GPT-4o-class baseline.
- `streamguardbench_stream.py`. StreamGuardBench under
  `aisafepy.stream`. Target: p95 added latency ≤ 80 ms with cascade enabled.
- `harmbench_stream.py`. HarmBench safety/utility plot for the cascade
  pipeline.
- `ailuminate_adapt.py`. AILuminate scenarios processed through
  `aisafepy.adapt`'s compiler, measuring time-to-fix reduction vs a
  hand-written Guardrails-AI validator baseline.

Each harness is a thin driver script (≤ 200 LOC) plus a config file
documenting the model, hardware, and dataset version. They are
intentionally **not** unit tests. They assume GPU access and downloaded
weights, and they're skipped in the default `pytest` run.
