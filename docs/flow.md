# `aisafepy.flow`: capability-based information-flow control

`flow` makes indirect prompt injection a *type error* rather than a
classification problem. Untrusted data is wrapped in `Tainted[T]` values
that carry labels; tools declare what they require; a policy declares what
is forbidden; and the interpreter checks every tool call against both.
This guide covers the concepts, the policy API (fluent and YAML), the
restricted plan evaluator, the framework-free adapter, the audit
recorder, and the linter.

## Concepts

### Labels

Every `Tainted` value carries three labels:

| Label | Type | Joined by | Meaning |
|---|---|---|---|
| `provenance` | `frozenset[str]` | union | Which sources the value (transitively) came from. |
| `capabilities` | `frozenset[str]` | union | What kind of data it may contain (`read.user`, `read.secrets`, ...). |
| `integrity` | `TRUSTED` > `UNTRUSTED` > `QUARANTINED` | meet (worst wins) | How much the planner may rely on it. |

Operations on tainted values propagate labels: `a + b`, `a * n`,
slicing, iteration, `tainted_join`, `collect`, and `map_values` all
produce values labelled with the join of their inputs. `lower(x)` strips
labels and is the explicit, greppable escape hatch; `str(x)` also lowers
and is documented as such.

Plain Python values are treated as `TRUSTED` with empty labels. That is
the right default for literals the developer wrote, and the wrong one for
strings the model produced; adapters therefore lift model output as
`UNTRUSTED` with provenance `llm.output`.

### The lattice laws

Label joining is a semilattice: commutative, associative, idempotent, and
monotone (joining can only lower integrity). `tests/test_flow_taint_extra.py`
checks these properties over a table of labels, so a refactor of
`taint.py` that breaks them fails CI.

### Checks at a call site

`evaluate_call(tool, args, kwargs, policy, metadata, context)` joins the
labels of every argument and checks, in order:

1. Control-flow integrity of the context against `require(control_flow_integrity=...)`.
2. Joined argument integrity against `require(arg_max_integrity=...)`.
3. Joined argument capabilities against the requirement's capability ceiling.
4. Argument integrity against `@secure_tool(required_integrity=...)`.
5. Every `deny_if` rule registered for the tool.

The first failure produces an `IFCViolation` carrying the joined labels
and the context's taint chain. In `strict` mode it is raised as
`IFCViolationError`; in `mediated` mode the adapter's mediator may approve
it, which is recorded as a separate allow decision.

## Building a policy

```python
from aisafepy.flow import Policy, Capability

policy = (
    Policy()
    .label_source("user_prompt", integrity="TRUSTED")
    .label_source("gmail.read", integrity="UNTRUSTED", caps={Capability.READ_USER})
    .label_source("vault.get", caps={Capability.READ_SECRETS})
    .require("send_email", control_flow_integrity="TRUSTED",
             caps={Capability.READ_USER})
    .require("payments.transfer", control_flow_integrity="TRUSTED",
             arg_max_integrity="TRUSTED")
    .deny_if("send_email",
             when={"capability_present": "read.secrets"},
             reason="secret-to-external-sink")
    .declassify("extract_amount", by="quarantined model, numeric JSON schema")
)
```

`deny_if(when=...)` accepts either a Python callable `(*args, **kwargs) -> bool`
or a declarative predicate (next section). Callables are the most
expressive; predicates are the ones that survive a round trip through YAML.

Two query helpers matter for adapters:

- `policy.output_label(tool)` returns the label a tool's *return value*
  should carry: the source label registered under the tool's name, else
  `TRUSTED` for a declassifier, else `UNTRUSTED` under `tool:<name>`.
- `policy.lift_output(tool, value)` applies it. A declassifier upgrades an
  already-tainted value to `TRUSTED` and records the justification in the
  value's `annotations`.

## YAML policies

```yaml
name: ops-assistant
mode: strict
sources:
  user_prompt: {integrity: TRUSTED}
  gmail.read:  {integrity: UNTRUSTED, capabilities: [read.user]}
  vault.get:   {capabilities: [read.secrets]}
requirements:
  send_email:
    control_flow_integrity: TRUSTED
    capabilities: [read.user]
  payments.transfer:
    control_flow_integrity: TRUSTED
    arg_max_integrity: TRUSTED
deny_rules:
  - tool: send_email
    reason: secret-to-external-sink
    when: {capability_present: read.secrets}
  - tool: send_email
    reason: external-recipient-from-untrusted-data
    when:
      all_of:
        - {arg: to, arg_integrity_below: TRUSTED}
        - {arg: to, value_matches: "@(?!corp\\.example$)"}
declassifications:
  extract_amount: {by: "quarantined model, numeric JSON schema"}
```

```python
policy = Policy.from_yaml("examples/flow_policy.yaml")
policy.to_yaml("policy.lock.yaml")        # or Policy.from_dict / to_dict
```

### Predicate vocabulary

| Operator | Matches when |
|---|---|
| `provenance_contains: name` or `[names]` | any argument's provenance intersects the names |
| `capability_present: cap` or `[caps]` | any argument carries one of the capabilities |
| `arg_integrity_below: LEVEL` | any argument's integrity is strictly worse than `LEVEL` |
| `value_matches: regex` | any string argument (tainted or plain) matches (`re.search`) |
| `any_of: [p, ...]`, `all_of: [p, ...]`, `not: p` | boolean combinators |

Add `arg: name` (keyword) or `arg: 0` (positional index) to a leaf to scope
it to one argument. A missing argument never matches. Malformed predicates
raise `PredicateError` at load time, not at call time.

Rules written as Python callables serialize as reason-only entries. By
default `policy_from_dict` / `from_yaml` refuse to load such a document
(`SerializationError`) because the rule cannot be reconstructed; pass
`strict=False` to drop those rules deliberately. `to_yaml` appends a
comment block naming every rule the file does *not* capture.

## Plans: the restricted DSL

`aisafepy.flow.dsl` evaluates a JSON action sequence emitted by a planner.
This is the CaMeL shape: the privileged model writes a small program from
the user's request; a sandboxed interpreter runs it; untrusted tool output
is data that the program may route but never execute.

```python
from aisafepy.flow.dsl import run_plan, validate_plan

plan = [
    {"call": "gmail.read", "args": {"query": "latest invoice"}, "bind": "mail"},
    {"call": "crm.lookup", "args": {"vendor": {"$ref": "vendor"}}, "bind": "crm"},
    {"call": "payments.transfer",
     "args": {"account": {"$get": [{"$ref": "crm"}, "account"]},
              "amount":  {"$get": [{"$ref": "crm"}, "amount_due"]}}},
]
trace = run_plan(plan, tools=TOOLS, policy=policy,
                 initial_bindings={"vendor": "Acme"}, unwrap_args=True)
```

### Step and expression grammar

A step: `{"call": tool, "args": {...} | [...], "bind": name, "if": expr, "note": text}`.

| Expression | Value | Labels |
|---|---|---|
| JSON literal | itself | none (nested containers join their elements) |
| `{"$ref": "x"}` | the binding `x` | `x`'s labels |
| `{"$get": [expr, key]}` | `expr[key]` | inherited, joined with a tainted key |
| `{"$concat": [e1, e2, ...]}` | string concatenation | join of all parts |
| `{"$format": "...{k}...", "with": {"k": expr}}` | `str.format` | join of all substituted values |
| `{"$lit": value}` | `value` verbatim, even if it contains `$` keys | none |

`validate_plan(plan, tools)` performs the static checks without executing:
unknown tools and keys, malformed expressions, references to names bound
later or never, duplicate binds, nesting depth, plan length. `run_plan`
calls it first and raises `PlanError` on any problem.

### Two flows, two checks

Each step is evaluated through `evaluate_call` with its arguments'
labels (data flow) and a control-flow integrity computed from the plan's
starting context and the labels of the step's `if` condition (control
flow). This separation is what lets a legitimate "read the email, forward
its subject to the owner" plan complete while both injection shapes fail:

- **Data-driven hijack.** `transfer(account="ACC-EVIL", amount=mail.body)`.
  The amount is `UNTRUSTED`; `arg_max_integrity="TRUSTED"` denies it.
- **Control-driven hijack.** `transfer(account=literal, amount=literal)`
  guarded by `"if": mail.subject`. Every argument is trusted, but the
  decision to call depended on the email; `control_flow_integrity="TRUSTED"`
  denies it.

`cfi_mode="data"` makes every argument dependency lower control-flow
integrity too. It is the pessimistic rule the simpler adapters use and is
appropriate when the tool must never run in the presence of untrusted
data at all; most policies express that more precisely with
`arg_max_integrity` and capabilities.

### Traces

`run_plan` returns a `PlanTrace`: one `StepResult` per step (`allow`,
`block`, `skipped`, `error`), the control-flow integrity used, the
decision or violation, a snapshot of every binding's labels, and timing.
On denial in strict mode the partial trace is attached to the raised
`IFCViolationError` as `.trace`; if a tool itself raises you get
`PlanToolError` with the same attribute. `trace.to_dict()` is JSON-ready
and redacts evidence. See `examples/flow_plan_dsl.py` for a complete run.

## Framework-free adapter

`aisafepy.flow.adapters.generic` wraps plain callables, which is what most
hand-rolled agent loops have:

```python
from aisafepy.flow.adapters.generic import ToolRegistry, secure_callables

registry = ToolRegistry()

@registry.register(name="send_email", capabilities={"read.user"})
def send_email(to: str, body: str) -> str:
    """Send an email on the user's behalf."""
    ...

tools = registry.secured(policy)          # {"send_email": <wrapped>}
schemas = registry.schemas()              # JSON-schema style, deterministic
```

Wrapped tools lift plain string arguments as `UNTRUSTED` / `llm.output`
for the check, call the inner function with the original arguments, and
label the return value with `policy.lift_output`. Sync and async callables
both work. `schemas()` derives parameter schemas from signatures and type
hints (including `Tainted[T]`, `Optional`, `Literal`, lists and dicts) and
is stable enough to hash with `aisafepy.flow.adapters.mcp.schema_hash` for
manifest pinning.

## Audit recorder

Every evaluation reports to `IFCContext.on_decision`. `DecisionRecorder`
is the built-in observer:

```python
from aisafepy.flow.audit import DecisionRecorder
from aisafepy.flow.interpreter import ifc_run

recorder = DecisionRecorder(path="flow-audit.jsonl", max_entries=5000)
with ifc_run(policy, recorder=recorder) as ctx:
    ...  # pass ctx to evaluate_call / run_plan / secure_callables
print(recorder.summary())
```

Records are flattened (`tool`, `kind`, `action`, `reason`, labels, ids,
taint chain) and evidence is redacted with the same function the
telemetry layer uses. Backends: ring buffer (always), JSONL (flushed on
run exit or immediately with `autoflush=True`), and a callback. Hook
exceptions are logged and never alter enforcement.

## Linting a policy

```python
from aisafepy.flow.lint import lint_policy, format_findings, has_errors

findings = lint_policy(policy, tools=registry.names())
print(format_findings(findings))
assert not has_errors(findings)
```

| Code | Severity | Why it matters |
|---|---|---|
| `deny-rule-raises` | error | A rule that raises is treated as "no match" at runtime and silently allows. The linter probes each rule with signature-shaped `Tainted` values. |
| `uncovered-tool` | warning | A tool the agent exposes that no requirement, deny rule, or declassifier mentions. |
| `rule-for-unknown-tool` | warning | Policy references a tool the agent does not have (typo or stale policy). |
| `deny-without-require` | warning | Deny rules exist but integrity and capabilities are unchecked. |
| `permissive-sink` | warning | A sink-looking tool whose requirement enforces nothing. |
| `weak-declassification` | warning | Empty or very short `by` justification. |
| `mediated-without-requirements` | warning | Nothing can reach the mediator. |
| `unused-source` | info | Labelled source nothing reacts to. |
| `capability-never-granted` | info | A requirement's ceiling names a capability no source grants. |
| `duplicate-deny-reason` | info | Same reason registered twice on one tool. |

## Limitations

- Labels are tracked at tool boundaries and through the DSL's expression
  language. Code *inside* a tool can mix data however it likes; tools that
  need to preserve labels should accept and return `Tainted` values.
- `Tainted.__bool__` is an implicit flow. Branching on a tainted value in
  ordinary Python is invisible to the policy; use the DSL's `if` when the
  branch matters.
- Declassification is trust in a tool, not a proof about it. The linter
  insists on a written justification because that is all the runtime can
  check.
- The DSL has no loops, no arbitrary expressions, and no tool-name
  indirection on purpose. If a planner needs those, it should emit several
  plans and let the host decide between them.
- See `docs/CAVEATS.md` for the research-level caveats (token overhead,
  obfuscation attacks against the planner, and so on).
