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
