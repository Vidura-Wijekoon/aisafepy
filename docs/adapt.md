# `aisafepy.adapt`: the eval-to-guardrail compiler

`adapt` closes the loop between red-team evidence and runtime guards.
Failures come in from PyRIT, Garak, Inspect, Promptfoo, DeepEval, a
tracing backend or a plain JSONL/CSV export. They are clustered,
gated on attack-success rate, compiled into runtime artifacts, written
to a bundle, canary-deployed with a false-positive budget, and every
step is recorded in a hash-chained audit log.

```
sources ──> cluster ──> ASR gate ──> targets ──> CompilationReport
                                                   │ save()/load()
                                                   ▼
                      evaluate_guard() <── CanaryController ──> GuardPipeline
                                                   │ record()/check()
                                                   ▼
                                               AuditLog
```

## 1. Sources

Every source yields `FailureRecord` objects:

| field | meaning |
|---|---|
| `id` | stable identifier from the source (or synthesized) |
| `input`, `output` | the attack text and the model's response |
| `was_violation` | the label the compiler trains and measures against |
| `attack_category` | free-form tactic label used for per-category metrics |
| `severity`, `scorer`, `score`, `timestamp` | optional provenance |
| `source` | `pyrit`, `garak`, `inspect`, `langfuse`, `promptfoo`, `deepeval`, `jsonl`, `csv` |
| `metadata` | the full native record, untouched |

| source | native format | notes |
|---|---|---|
| `PyRITSource(memory_db= / jsonl_path=)` | PyRIT DuckDB or JSONL memory | `only_violations=True` by default |
| `GarakReport(path)` | `report.jsonl` | `attempt` / `digest` entries |
| `InspectLog(path)` | `*.eval` JSON or JSONL | |
| `LangfuseTraces(client, filter=)` | duck-typed `fetch_traces` | alias `ProductionTraces` |
| `PromptfooSource(path)` | `promptfoo eval -o out.json` | failed test = violation; `metadata.pluginId` is the category |
| `DeepEvalSource(path)` | DeepEval test-run JSON | any failed metric = violation |
| `JSONLSource(path, mapping=)` | any JSON Lines | dotted-path field mapping, see below |
| `CSVSource(path, mapping=)` | CSV with header | same mapping rules |
| `InMemorySource(records)` | Python list | tests, notebooks |

### Field mapping

`JSONLSource` and `CSVSource` map `FailureRecord` fields to paths in
the native record. Unmapped fields fall back to a list of common names
(`DEFAULT_MAPPING`), and the first path that resolves wins:

```python
JSONLSource("qa.jsonl", mapping={
    "input": "request.prompt",
    "output": "response.choices.0.text",
    "was_violation": ["judge.unsafe", "flags.blocked"],
})
```

Boolean-ish values are parsed uniformly: `true`, `yes`, `1`, `unsafe`,
`fail`, `violation` (case-insensitive) and non-zero numbers are
violations.

### Combining and filtering

```python
from aisafepy.adapt.sources import merge_sources, dedupe_records, RecordFilter

records = dedupe_records(merge_sources(pyrit, garak, promptfoo))
recent_jailbreaks = list(
    RecordFilter().categories("jailbreak").min_severity("high").since(cutoff).apply(records)
)
```

`merge_sources` keeps ids unique by prefixing collisions with the
source name. `dedupe_records` compares case- and whitespace-folded
`(input, output)` by default.

## 2. Compiling

```python
from aisafepy.adapt import GuardCompiler, Target
from aisafepy.adapt.compile import CompilerHooks

compiler = GuardCompiler(
    source=records,
    negatives=benign_records,                       # see below
    targets=[
        Target.synthesize_regex(min_precision=0.99),
        Target.policy_rule(dsl="cedar"),
        Target.deliberative_case(policy="policies/company_safety.md"),
    ],
    min_attack_success_rate=0.10,
    max_clusters=20,
    hooks=CompilerHooks(on_artifact=lambda a: print("compiled", a.name)),
)
report = compiler.compile()
print(report.to_markdown())
```

* **Clustering** runs over violations and non-violations together, so
  a cluster's attack-success rate (ASR) is meaningful. Clusters below
  `min_attack_success_rate` land in `report.rejected_clusters`; with
  `max_clusters` the largest survivors are kept.
* **Targets** receive each kept cluster. A target whose
  `compile_for_cluster` accepts a `negatives` keyword also receives the
  benign pool. A target that raises produces a `failed-<label>`
  artifact with the error in `metadata` instead of aborting the run.
* `dry_run=True` clusters and gates without compiling anything.
* `report.metadata` records counts, the target list, duration and any
  hook errors.

### Negatives and precision

A synthesized regex is only as good as its precision on benign
traffic. The compiler builds a benign pool from:

1. non-violation records in `source` (use `only_violations=False` on
   the source to keep them), and
2. the `negatives=` argument: `FailureRecord`s (non-violations are
   used) or plain strings.

The regex target then reports, per pattern, `precision = TP / (TP +
FP)` against that pool and `recall` against the cluster's positives.
Without any negatives it falls back to recall as a proxy and marks the
artifact `precision_source="recall_proxy"`; treat such artifacts as
shadow-only candidates.

Synthesized patterns are word-aligned n-grams with digit runs folded to
`\d+` and whitespace to `\s+`, bounded by `\b`. Every pattern must pass
`safe_regex()` (no nested or stacked quantifiers) and is applied with
`timed_search()`, which caps the input length, so a compiled Tier-1
guard can never become a regex-DoS vector.

### Report bundle

```python
report.save("guards/2026-11-10")      # directory bundle
report = CompilationReport.load("guards/2026-11-10")
```

```
guards/2026-11-10/
├── report.json                  # self-contained: artifacts with payloads, cluster stats
├── report.md                    # human summary
├── regex-3.regex.json           # one sidecar per artifact payload
├── cedar-rule-cluster-3.cedar
├── rego-rule-cluster-7.rego
└── deliberative-3.md
```

Deferred artifacts (classifier / steering when their training deps are
missing) keep their full training spec in `metadata` and have no
sidecar. Cluster records are not serialized by default; pass
`include_records=True` to `save()` / `to_dict()` when the bundle must
be reproducible on its own.

## 3. Canary deployment

```python
from aisafepy.adapt.canary import CanaryController

ctrl = CanaryController(report, pipeline,
                        canary_traffic_pct=0.10, fp_budget=0.005,
                        min_samples=200, state_dir="guards/canary")
ctrl.start(shadow=True, shadow_in_tier=True)   # evaluate, never enforce
```

| method | what it does |
|---|---|
| `start(shadow=, shadow_in_tier=)` | materialize artifacts into guards and attach them live or in shadow |
| `route(request_key)` | deterministic SHA-256 bucket test; the same key is always in or out |
| `record(decision, labeled_benign=, request_key=)` | account one decision to a managed guard |
| `fp_rate(guard_name=None)` | FP rate over labeled-benign samples, per guard or pooled |
| `check()` | roll back guards over budget (after `min_samples`), persist, return a `CanaryResult` |
| `promote(name)` / `promote_all(require_fp_ok=True)` | shadow to live |
| `rollback(name)` | remove a guard from every tier and the shadow slot |
| `state()` / `save()` / `CanaryController.load(dir, report, pipeline)` | persistence |

`promote(report, to=pipeline, ...)` from the README builds a controller,
starts it, runs one `check()` and leaves the controller on
`pipeline.__aisafepy_canary__`.

### FP budget math

Only requests the caller labels benign (`labeled_benign=True`) count
toward the estimate. A block or escalate on a benign request is one
false positive:

```
fp_rate(guard) = benign_fp / benign_samples
rollback when  benign_samples >= min_samples and fp_rate > fp_budget
```

Unlabeled production traffic increases `samples` and `flag_rate` but
never triggers a rollback; `check()` says so in its notes. Choose
`min_samples` so that one false positive cannot by itself breach the
budget (`min_samples > 1 / fp_budget`), otherwise the first mistake on
a fresh canary rolls it back.

With `shadow_in_tier=True` the controller inserts `ShadowGuard`
wrappers into the pipeline tiers. A wrapper runs the real guard,
records the result (reading `labeled_benign` and `request_key` from
`Context.metadata`) and returns an `allow` whose evidence carries the
suppressed action, so shadow candidates see live traffic with zero
user impact.

## 4. Offline evaluation

```python
from aisafepy.adapt.evaluate import evaluate_guard, threshold_sweep, best_threshold

rep = evaluate_guard(pipeline, labeled_records, field="input")
print(rep.to_markdown())          # confusion matrix, per-category, latency, misses

sweep = threshold_sweep(lambda t: HFClassifierGuard.from_hf(MODEL, threshold=t),
                        labeled_records, [0.3, 0.5, 0.7, 0.9])
t, metrics = best_threshold(sweep, max_fpr=0.01)
```

`evaluate_guard` accepts a single guard (sync or async) or a whole
`GuardPipeline`. By default a record is "flagged" when the action is
block or escalate; pass `threshold=` to flag on the decision's unsafe
score instead (block confidence, or `1 - confidence` for allows).
Guard exceptions are counted in `errors` and treated as not flagged.

## 5. Governance

```python
from aisafepy.adapt.governance import AuditLog

log = AuditLog("audit/compiler.jsonl", hmac_key=os.environb[b"AISAFEPY_AUDIT_KEY"])
log.append("compile", actor="ci", payload={"bundle": "guards/2026-11-10"})
log.append_decision(decision, actor="pipeline")      # redacted automatically
res = log.verify_detailed()
if not res.ok:
    alert(f"audit chain broken at entry {res.first_bad_index}: {res.reason}")
```

Each line is `{...entry, "__hash__": sha256(entry), "__sig__": hmac(key, __hash__)}`;
`prev_hash` links to the previous line, so edits, insertions and
deletions all surface as the first bad index. Appends take an OS file
lock (`flock` on POSIX, a sentinel-byte `msvcrt` lock on Windows) and
re-read the tail inside the lock, so several processes can append to
one log without forking the chain. `tail(n)`, `iter_entries()` and
`export_csv()` cover the common read paths.
