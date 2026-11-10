# Benchmarks

Two kinds of benchmark live here:

- **Offline benchmarks** run with only the core install, no GPU, no
  network, and no model downloads. They are smoke-tested in CI
  (`tests/test_benchmarks_smoke.py`) and are the ones you can run today.
- **Planned GPU harnesses** reproduce the external leaderboards from the
  design proposal. They require downloaded weights and accelerator
  access and are intentionally not part of the default `pytest` run.

## Offline benchmarks

Each script accepts `--json` (machine-readable output) and `--quick`
(tiny sizes, used by the smoke test). Run from the repository root.

| Script | Question it answers |
|--------|---------------------|
| `bench_tier1_latency.py` | How fast are the Tier-1 guards? p50/p95/p99 and throughput for the PII regex set, a 1000-term Aho-Corasick blocklist, and a three-guard pipeline, over 1000 texts including a 50 KB stressor. |
| `bench_detection.py` | How much does Tier-1 alone catch? Per-family recall plus overall precision, recall, F1, and false-positive rate against `datasets/injection_eval.jsonl`. Add a Tier-2 guard with `--guard module:factory`. |
| `bench_flow_overhead.py` | What does IFC enforcement cost per tool call? Microseconds for `evaluate_call` with 0, 1, and 5 deny rules versus a bare call. |
| `bench_budgets.py` | How fast is `Budget.consume`, single-threaded and under contention? |

```bash
python benchmarks/bench_tier1_latency.py
python benchmarks/bench_detection.py
python benchmarks/bench_detection.py --guard mypkg.guards:make_classifier
python benchmarks/bench_flow_overhead.py --json
python benchmarks/bench_budgets.py
```

### The dataset

`datasets/injection_eval.jsonl` is a labelled evaluation set of at least
160 records. Each line is `{"id", "text", "label", "family", ...}` where
`label` is `attack` or `benign`. Attack families cover instruction
override, role hijack, system-prompt exfiltration, delimiter injection,
tool poisoning, encoded payloads (base64 / hex / rot13, carrying a
`decodes_to` field verified in CI), obfuscation (zero-width and
homoglyph), many-shot priming, and markdown-image data exfiltration.
Benign records are deliberately hard negatives: support tickets and
docs that quote "ignore previous instructions", recipes with "steps",
fiction, code-review comments, security-blog excerpts that quote
attacks, legitimate financial requests, and multilingual text.

The dataset exists to make Tier-1's shallowness measurable. Literal
deterministic guards will show high precision and low recall on the
encoded and obfuscated families. That gap is the argument for the
`stream` cascade's Tier-2 classifier, not a defect in the dataset.

### Results format

Record results in a table like this when you run on a fixed machine:

| Benchmark | Metric | Value | Machine | Date |
|-----------|--------|-------|---------|------|
| tier1_latency | pipeline p95 ms | _tbd_ | _cpu model_ | _yyyy-mm-dd_ |
| detection | precision / recall / FPR | _tbd_ | _n/a_ | _yyyy-mm-dd_ |
| flow_overhead | us/call, 1 deny rule | _tbd_ | _cpu model_ | _yyyy-mm-dd_ |
| budgets | consume/s, 8 threads | _tbd_ | _cpu model_ | _yyyy-mm-dd_ |

Numbers are hardware dependent. Always capture the CPU model and the
Python version alongside them.

## Planned (GPU) harnesses

These reproduce the benchmarks called out in the proposal. They assume
GPU access and downloaded weights and are not wired into `pytest`.

- `agentdojo_flow.py`. AgentDojo on top of `aisafepy.flow`. Target:
  attack success rate <= 5% with >= 90% utility against a GPT-4o-class
  baseline.
- `streamguardbench_stream.py`. StreamGuardBench under
  `aisafepy.stream`. Target: p95 added latency <= 80 ms with the cascade
  enabled.
- `harmbench_stream.py`. HarmBench safety/utility plot for the cascade
  pipeline.
- `ailuminate_adapt.py`. AILuminate scenarios through `aisafepy.adapt`'s
  compiler, measuring time-to-fix reduction versus a hand-written
  Guardrails-AI validator baseline.

Each is a thin driver script plus a config file documenting the model,
hardware, and dataset version.
