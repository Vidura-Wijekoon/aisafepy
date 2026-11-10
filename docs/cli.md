# CLI reference

The `aisafepy` command is installed with the package. It uses only the
core dependencies, so it works in any environment where `import aisafepy`
works, with no torch or agent framework required.

## Global behaviour

```
aisafepy [--json] [--quiet] [--version] COMMAND ...
```

- `--json` emits machine-readable JSON instead of a table, everywhere it
  is meaningful.
- `--quiet` / `-q` suppresses informational lines; errors and flagged
  findings still print.
- `--version` prints the installed version.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | completed, nothing flagged |
| 1 | usage error, unreadable input, or an unexpected failure |
| 2 | completed but something was flagged: a guard blocked, a chain failed to verify, a manifest drifted |

Exit code 2 is what you branch on in CI and pre-commit hooks. The
command never raises an exception to the shell; failures are printed on
one line.

## `aisafepy scan`

Run Tier-1 deterministic guards over files, directories, or stdin.

```
aisafepy scan [PATH ...] [--pii] [--redact]
              [--blocklist TERM ...] [--blocklist-file FILE]
              [--pattern NAME=REGEX ...] [--pattern-action block|transform]
              [--max-chars N] [--include GLOB ...] [--no-default-ignores]
              [--format table|json|jsonl] [--fail-on block|escalate|transform|never]
              [--show-allowed]
```

With no guard flags the built-in PII set is used. Directories are walked
recursively, skipping `.git`, `node_modules`, `__pycache__`, `.venv`, and
similar; binary and non-UTF-8 files are skipped with a note. `-` reads
stdin.

```bash
echo "my ssn is 123-45-6789" | aisafepy scan          # exit 2
aisafepy scan --pii --redact README.md                # transform, exit 0
aisafepy scan --blocklist "api_key=" src/             # exit 2 on a hit
aisafepy scan --pattern jira='JIRA-\d+' --format jsonl notes.txt
aisafepy scan --pii --include '*.py' --json .
```

## `aisafepy policy`

Evaluate a runtime `PolicyDocument` YAML.

```
aisafepy policy check  POLICY.yaml (--input TEXT | --input-file FILE) [--role R] [--field KEY=VALUE ...]
aisafepy policy explain POLICY.yaml (--input TEXT | --input-file FILE) [--role R] [--field KEY=VALUE ...]
aisafepy policy show    POLICY.yaml
```

`check` reports the first matching rule and exits 2 when its decision is
block or escalate. `explain` shows every rule and marks the winner.
`show` dumps the loaded document with each rule's selector kind.

```bash
aisafepy policy check policies/example_runtime_policy.yaml --input "ignore previous instructions"
aisafepy policy explain policies/example_runtime_policy.yaml --input "email bob@example.org"
aisafepy policy show policies/example_runtime_policy.yaml --json
```

## `aisafepy audit`

Operate on the hash-chained `AuditLog`.

```
aisafepy audit verify LOG.jsonl
aisafepy audit tail   LOG.jsonl [-n N] [--event E] [--actor A]
aisafepy audit append LOG.jsonl --event E --actor A [--payload JSON]
aisafepy audit stats  LOG.jsonl
```

`verify` walks the SHA-256 chain and exits 2 on the first break, naming
the line. `append` extends the chain (creating the log if missing) and
is meant for scripting governance events.

```bash
aisafepy audit verify guards/audit.jsonl
aisafepy audit tail guards/audit.jsonl -n 5 --actor ops
aisafepy audit append guards/audit.jsonl --event deploy --actor ci --payload '{"env":"prod"}'
aisafepy audit stats guards/audit.jsonl --json
```

## `aisafepy manifest`

Pin and verify MCP tool manifests.

```
aisafepy manifest pin    TOOLS.json [-o MANIFEST.json] [--strict] [--extra-pattern REGEX ...]
aisafepy manifest verify TOOLS.json MANIFEST.json [--allow-added] [--allow-removed]
aisafepy manifest diff   A.json B.json
```

`TOOLS.json` is a `tools/list` result or a list of
`{name, description, inputSchema}` objects. `pin` hashes each tool's
schema and screens descriptions for hidden-instruction patterns;
`--strict` refuses to write when a description looks poisoned. `verify`
exits 2 on drift, additions, or removals.

```bash
aisafepy manifest pin tools.json -o manifest.json
aisafepy manifest verify tools.json manifest.json       # exit 2 on drift
aisafepy manifest pin tools.json --strict               # exit 2 on screening hits
```

## `aisafepy compile`

Compile red-team failures into guard artifacts, offline by default.

```
aisafepy compile --source pyrit-jsonl|garak|inspect|jsonl PATH
                 [--target regex] [--target policy[:cedar|rego]]
                 [--target deliberative:POLICY.md] [--target classifier[:BASE]]
                 [--min-asr F] [--min-cluster-size N] [--all-records] [--online] [-o DIR]
```

Clustering is offline TF-IDF by default so the command is safe in CI;
`--online` opts into the full `GuardCompiler` with sentence-transformers.
With `-o DIR` it writes `report.txt`, `report.json`, and one file per
artifact payload.

```bash
aisafepy compile --source pyrit-jsonl runs.jsonl --target regex --target policy:rego -o guards/
aisafepy compile --source jsonl records.jsonl --min-asr 0.2 --json
```

## `aisafepy decisions`

Summarise or filter a JSONL log of `GuardDecision` records.

```
aisafepy decisions summarize DECISIONS.jsonl [--top N]
aisafepy decisions filter    DECISIONS.jsonl [--action A ...] [--severity S ...]
                             [--guard NAME ...] [--tier N ...]
                             [--min-confidence F] [--max-confidence F] [--min-latency-ms F]
                             [--contains TEXT] [--limit N] [--count]
```

`summarize` reports counts by action, severity, guard, and tier, latency
percentiles, and the top rationales. `filter` re-emits matching records
as JSONL (or just a count with `--count`). Both accept `-` for stdin.

```bash
aisafepy decisions summarize decisions.jsonl
aisafepy decisions filter decisions.jsonl --action block --min-confidence 0.9 > blocks.jsonl
cat decisions.jsonl | aisafepy --json decisions summarize -
```

## `aisafepy pipeline`

Work with guard cascades declared in YAML or JSON (`aisafepy.stream.config`).

| Subcommand | What it does | Exit 2 when |
|---|---|---|
| `pipeline validate CONFIG.yaml` | Static checks: unknown kinds, missing required parameters, schema errors. Nothing heavy is built. | any problem is found |
| `pipeline describe CONFIG.yaml` | Builds the pipeline and prints `GuardPipeline.describe()`; `--json` prints `to_dict()`. | never |
| `pipeline run CONFIG.yaml --input TEXT` | Evaluates one input and prints action, severity, tier, guard, confidence, rationale, evidence. | the decision is block or escalate |
| `pipeline registry` | Lists every guard kind the default registry accepts and its required parameters. | never |
| `pipeline bench CONFIG.yaml --input-file F` | Replays the file `--repeat` times; prints p50 / p95 / p99 / max / mean latency, action counts and per-guard metrics. | never |

`run` options: `--input TEXT` or `--input-file FILE` (`-` for stdin),
`--role user|assistant|tool` (default `assistant`), `--prompt TEXT` for
guards that use the prompt, `--metadata KEY=VALUE` (repeatable, lands
in `Context.metadata`). With `--json` the decision is printed as a
redacted `GuardDecision` under `decision`, plus `flagged`.

```bash
aisafepy pipeline validate guards.yaml
aisafepy pipeline run guards.yaml --input "ignore all previous instructions" --role user
aisafepy --json pipeline run guards.yaml --input-file - < response.txt
aisafepy pipeline bench guards.yaml --input-file samples.txt --repeat 20
aisafepy pipeline registry
```

## `aisafepy flow`

Work with IFC policies (`aisafepy.flow`), planner-emitted action plans
and `DecisionRecorder` output.

| Subcommand | What it does | Exit 2 when |
|---|---|---|
| `flow lint POLICY.yaml [--tools a,b]` | Runs `lint_policy`; `--tools` enables coverage findings; `--strict` also fails on warnings. | any error (or warning with `--strict`) |
| `flow show POLICY.yaml` | Prints sources, requirements, deny rules and declassifiers as tables; `--json` prints `policy_to_dict()`. | never |
| `flow plan POLICY.yaml PLAN.json --dry-run` | `validate_plan` without executing. | the plan is invalid |
| `flow plan POLICY.yaml PLAN.json --tools SPEC[,SPEC]` | Executes the plan with `run_plan` and prints the trace (step, tool, action, control-flow integrity, binding, detail). | the plan halts on a violation |
| `flow audit RECORDS.jsonl` | Summarises a recorder file: counts per action and tool, mediated approvals, top deny reasons (`--top N`). | never |

Tool specs are `module:attr` (the callable's name becomes the tool
name) or `name=module:attr` when the tool name differs from the Python
name, for example `gmail.read=mytools:gmail_read`. `--cfi` sets the
starting control-flow integrity (default `TRUSTED`); `--cfi-mode
control|data` selects how the evaluator derives per-step integrity (see
`docs/flow.md`). A plan file is a JSON list of steps or an object with
a `steps` list.

```bash
aisafepy flow lint policy.yaml --tools gmail.read,send_email,payments.transfer --strict
aisafepy flow show policy.yaml
aisafepy flow plan policy.yaml plan.json --dry-run
aisafepy flow plan policy.yaml plan.json \
    --tools gmail.read=agent.tools:gmail_read,payments.transfer=agent.tools:transfer
aisafepy flow audit runs/decisions.jsonl --top 5
```

## `aisafepy eval`

Measure a guard or a pipeline against a labelled JSON Lines dataset
with `aisafepy.adapt.evaluate`. Defaults match
`benchmarks/datasets/injection_eval.jsonl`.

```
aisafepy eval DATASET.jsonl (--guard module:factory[:kwargs-json] | --pipeline CONFIG.yaml)
                            [--text-field text] [--label-field label] [--positive-label attack]
                            [--family-field family] [--id-field id] [--role user]
                            [--flag-actions block,escalate] [--threshold-sweep 0.1,0.3,0.5]
                            [--min-recall R] [--max-fpr F] [--max-misclassified N]
```

* `--guard` accepts a class, a factory function, or a ready-made guard
  instance; an optional third segment is a JSON object of keyword
  arguments, for example `aisafepy.stream.injection:InjectionHeuristicGuard:{"block_threshold":0.7}`.
* `--threshold-sweep` re-instantiates the factory with `threshold=T`
  for each value and prints precision, recall, F1 and FPR per threshold.
* `--min-recall` and `--max-fpr` make the command exit 2 when the gate
  is violated, which turns it into a CI check for guard regressions.
* Output lists precision, recall, F1, FPR, accuracy, the confusion
  counts, latency, a per-family table (recall and FPR per
  `attack_category`), and the ids of misclassified records. `--json`
  returns the `EvaluationReport` dictionary plus `gates` and `sweep`.

```bash
aisafepy eval benchmarks/datasets/injection_eval.jsonl \
    --guard aisafepy.stream.injection:InjectionHeuristicGuard
aisafepy eval benchmarks/datasets/injection_eval.jsonl \
    --guard aisafepy.stream.injection:InjectionHeuristicGuard \
    --threshold-sweep 0.5,0.7,0.85 --max-fpr 0.05
aisafepy --json eval data.jsonl --pipeline guards.yaml --min-recall 0.8
```
