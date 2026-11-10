"""A restricted evaluator for planner-emitted action sequences.

This is the ``aisafepy.flow.dsl`` module the interpreter promised. In
the CaMeL architecture a privileged planner emits a *program* over
tool calls rather than issuing calls one at a time, and a sandboxed
interpreter runs that program while tracking how data flows between
steps. The payoff is that control-flow integrity becomes something the
runtime computes from the data a step depends on, instead of a flag an
adapter has to guess.

The plan language is deliberately tiny and JSON-native so a model can
emit it as structured output and a human can read it in an audit log.

Plan format
-----------

A plan is a list of steps. Each step is a mapping::

    {"call": "gmail.read",  "args": {"query": "latest"}, "bind": "mail"}
    {"call": "summarize",   "args": {"text": {"$get": [{"$ref": "mail"}, "body"]}},
     "bind": "summary"}
    {"call": "send_email",  "args": {"to": "boss@corp.example",
                                     "body": {"$ref": "summary"}}}

Step keys: ``call`` (required tool name), ``args`` (mapping for keyword
arguments or list for positional; default empty), ``bind`` (name to
bind the result to), ``if`` (an expression; the step is skipped when
it is falsy), ``note`` (free text, ignored).

Argument expressions:

* JSON literals (``str``, numbers, ``bool``, ``null``, lists, mappings
  without ``$``-keys) evaluate to themselves. Nested lists / mappings
  are evaluated element-wise and the result carries the joined labels.
* ``{"$ref": "name"}``: a previously bound value.
* ``{"$get": [expr, key_or_index]}``: item access; labels inherited.
* ``{"$concat": [expr, ...]}``: string concatenation; labels joined.
* ``{"$format": "Hello {who}", "with": {"who": expr}}``: ``str.format``
  with every substituted value's labels joined.
* ``{"$lit": value}``: force ``value`` to be a literal (escape hatch for
  mappings that legitimately contain ``$``-prefixed keys).

Integrity model
---------------

Every tool output is lifted with :meth:`Policy.lift_output`, so a
value read from ``gmail.read`` carries that source's label and a
declassifier's output is ``TRUSTED``. Two flows are tracked per step:

* **Data flow.** Arguments carry labels; ``evaluate_call`` checks them
  against ``arg_max_integrity`` and the capability ceiling. A transfer
  whose amount was copied out of an email fails on argument integrity.
* **Control flow.** Which steps run is decided by the plan (authored by
  the planner from the user's request, so it inherits the starting
  context's integrity) and by each step's ``if`` condition. The
  evaluator computes a step's control-flow integrity as the meet of
  the starting CFI and the labels of its ``if`` expression. A transfer
  gated on a value read from an email fails on control-flow integrity
  even though its arguments are all literals. This mirrors CaMeL, where
  the program comes from the privileged planner and untrusted data can
  only influence control flow through explicit conditionals.

``cfi_mode="data"`` switches to the pessimistic rule some adapters use:
every value a step's arguments depend on also lowers its control-flow
integrity. That makes ``control_flow_integrity="TRUSTED"`` equivalent
to "no untrusted data anywhere in the call", which is safe but rejects
most legitimate summarise-and-forward plans; prefer expressing that
with ``arg_max_integrity`` and capabilities.

What this module does **not** do: it is not a Python sandbox, it has no
loops or arbitrary expressions, and it cannot see dependencies that
happen *inside* a tool. Those remain the tool author's responsibility.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aisafepy.core.decisions import GuardDecision, IFCViolation
from aisafepy.core.telemetry import redact
from aisafepy.flow.interpreter import IFCContext, IFCViolationError, evaluate_call
from aisafepy.flow.policy import Policy, ToolMetadata
from aisafepy.flow.taint import Integrity, Tainted, _meet, labels_of, lower

STEP_KEYS = frozenset({"call", "args", "bind", "if", "note"})
_OPERATORS = frozenset({"$ref", "$get", "$concat", "$format", "$lit"})
MAX_EXPR_DEPTH = 32


class PlanError(ValueError):
    """The plan is malformed or exceeded a limit.

    ``step`` is the zero-based index of the offending step when known.
    """

    def __init__(self, message: str, *, step: int | None = None):
        prefix = f"step {step}: " if step is not None else ""
        super().__init__(prefix + message)
        self.step = step


class PlanToolError(RuntimeError):
    """A tool raised while the plan was executing.

    The original exception is chained; ``trace`` holds the partial
    :class:`PlanTrace` up to and including the failing step.
    """

    def __init__(self, step: int, tool: str, exc: BaseException, trace: PlanTrace):
        super().__init__(f"step {step}: tool {tool!r} raised {type(exc).__name__}: {exc}")
        self.step = step
        self.tool = tool
        self.trace = trace


@dataclass(frozen=True)
class StepResult:
    """Outcome of one plan step."""

    index: int
    tool: str
    action: str
    """``allow``, ``block`` (violation), ``skipped`` (falsy ``if``), or ``error``."""
    cfi: Integrity
    decision: GuardDecision | None = None
    violation: IFCViolation | None = None
    bound: str | None = None
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Snapshot of every binding's labels *after* this step:
    ``{name: {"provenance": [...], "capabilities": [...], "integrity": ...}}``."""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "action": self.action,
            "cfi": self.cfi,
            "bound": self.bound,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "decision": (
                redact(self.decision.model_dump(mode="json")) if self.decision else None
            ),
            "violation": (
                _violation_dict(self.violation) if self.violation else None
            ),
            "labels": self.labels,
        }


@dataclass
class PlanTrace:
    """Everything that happened while running a plan."""

    steps: list[StepResult] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    """Final bindings (``Tainted`` values)."""
    halted_at: int | None = None
    violation: IFCViolation | None = None

    @property
    def completed(self) -> bool:
        return self.halted_at is None

    def result(self, name: str) -> Any:
        """Lowered value of a binding (convenience for callers / tests)."""
        return lower(self.bindings[name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "halted_at": self.halted_at,
            "violation": _violation_dict(self.violation) if self.violation else None,
            "steps": [s.to_dict() for s in self.steps],
            "bindings": {name: _label_dict(v) for name, v in self.bindings.items()},
        }


def _violation_dict(v: IFCViolation) -> dict[str, Any]:
    d = v.model_dump(mode="json")
    d["provenance"] = sorted(v.provenance)
    d["capabilities"] = sorted(v.capabilities)
    d["required_capabilities"] = sorted(v.required_capabilities)
    return d


def _label_dict(value: Any) -> dict[str, Any]:
    prov, caps, integrity = labels_of(value)
    return {
        "provenance": sorted(prov),
        "capabilities": sorted(caps),
        "integrity": integrity,
    }


# ---- static validation -----------------------------------------------------------


def validate_plan(
    plan: Any,
    tools: Iterable[str] | Mapping[str, Any],
    *,
    initial_bindings: Iterable[str] = (),
    max_steps: int = 64,
) -> list[str]:
    """Check a plan without executing it. Returns a list of problems.

    Catches: non-list plans, non-mapping steps, missing / non-string
    ``call``, unknown step keys, unknown tools, malformed ``args``,
    malformed expressions, references to names that are not bound by an
    earlier step (forward and self references included), duplicate
    binds, and plans longer than ``max_steps``.
    """
    problems: list[str] = []
    if not isinstance(plan, list):
        return [f"plan must be a list of steps (got {type(plan).__name__})"]
    if len(plan) > max_steps:
        problems.append(f"plan has {len(plan)} steps; max_steps is {max_steps}")
    known = set(tools)
    bound: set[str] = set(initial_bindings)
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            problems.append(f"step {i}: must be a mapping (got {type(step).__name__})")
            continue
        extra = sorted(set(step) - STEP_KEYS)
        if extra:
            problems.append(f"step {i}: unknown key(s) {extra}")
        call = step.get("call")
        if not isinstance(call, str) or not call:
            problems.append(f"step {i}: 'call' must be a non-empty tool name")
        elif call not in known:
            problems.append(f"step {i}: unknown tool {call!r}")
        args = step.get("args", {})
        if isinstance(args, dict):
            exprs: list[Any] = list(args.values())
            if any(not isinstance(k, str) for k in args):
                problems.append(f"step {i}: 'args' mapping keys must be strings")
        elif isinstance(args, list):
            exprs = list(args)
        else:
            problems.append(f"step {i}: 'args' must be a mapping or list")
            exprs = []
        if "if" in step:
            exprs.append(step["if"])
        for expr in exprs:
            for problem in _validate_expr(expr, bound, depth=0):
                problems.append(f"step {i}: {problem}")
        bind = step.get("bind")
        if bind is not None:
            if not isinstance(bind, str) or not bind:
                problems.append(f"step {i}: 'bind' must be a non-empty string")
            elif bind in bound:
                problems.append(f"step {i}: binding {bind!r} is already bound (rebinding is not allowed)")
            else:
                bound.add(bind)
    return problems


def _validate_expr(expr: Any, bound: set[str], *, depth: int) -> list[str]:
    if depth > MAX_EXPR_DEPTH:
        return [f"expression nesting exceeds {MAX_EXPR_DEPTH}"]
    if isinstance(expr, list):
        out: list[str] = []
        for e in expr:
            out += _validate_expr(e, bound, depth=depth + 1)
        return out
    if not isinstance(expr, dict):
        return []
    ops = [k for k in expr if isinstance(k, str) and k.startswith("$")]
    if not ops:
        out = []
        for v in expr.values():
            out += _validate_expr(v, bound, depth=depth + 1)
        return out
    if len(ops) != 1 or ops[0] not in _OPERATORS:
        return [f"invalid expression operator(s) {sorted(ops)}; expected one of {sorted(_OPERATORS)}"]
    op = ops[0]
    body = expr[op]
    allowed_keys = {op, "with"} if op == "$format" else {op}
    extra = sorted(set(expr) - allowed_keys)
    if extra:
        return [f"{op}: unexpected key(s) {extra}"]
    if op == "$ref":
        if not isinstance(body, str) or not body:
            return ["$ref expects a binding name"]
        return [] if body in bound else [f"$ref to unbound name {body!r}"]
    if op == "$lit":
        return []
    if op == "$get":
        if not isinstance(body, list) or len(body) != 2:
            return ["$get expects [expr, key_or_index]"]
        return _validate_expr(body[0], bound, depth=depth + 1)
    if op == "$concat":
        if not isinstance(body, list):
            return ["$concat expects a list of expressions"]
        out = []
        for e in body:
            out += _validate_expr(e, bound, depth=depth + 1)
        return out
    # $format
    if not isinstance(body, str):
        return ["$format expects a template string"]
    with_ = expr.get("with", {})
    if not isinstance(with_, dict):
        return ["$format 'with' must be a mapping"]
    out = []
    for v in with_.values():
        out += _validate_expr(v, bound, depth=depth + 1)
    return out


# ---- execution ----------------------------------------------------------------------


def run_plan(
    plan: list[dict[str, Any]],
    *,
    tools: Mapping[str, Callable[..., Any]],
    policy: Policy,
    context: IFCContext | None = None,
    initial_bindings: Mapping[str, Any] | None = None,
    source_for_tool: Mapping[str, str] | None = None,
    mediator: Callable[[IFCViolation], bool] | None = None,
    max_steps: int = 64,
    unwrap_args: bool = False,
    cfi_mode: str = "control",
) -> PlanTrace:
    """Execute ``plan`` under IFC enforcement and return its trace.

    Parameters
    ----------
    tools:
        Tool name to callable. Callables receive keyword arguments (or
        positional ones when the step's ``args`` is a list). By default
        they receive ``Tainted`` values; pass ``unwrap_args=True`` for
        tools written against plain values.
    policy:
        The IFC policy. ``policy.mode`` selects strict / mediated.
    context:
        Starting :class:`IFCContext`. Its ``control_flow_integrity`` is
        the floor for every step; its ``taint_chain`` and ``on_decision``
        hook are shared with every per-step evaluation.
    initial_bindings:
        Values available to ``$ref`` before the first step (e.g. the
        user's prompt). Plain values are wrapped as ``TRUSTED`` with
        provenance ``binding:<name>``; ``Tainted`` values keep labels.
    source_for_tool:
        Override the source label used for a tool's output (see
        :meth:`Policy.output_label`).
    mediator:
        Human-approval callback for mediated mode.
    max_steps:
        Upper bound on plan length.
    unwrap_args:
        Pass plain values (``lower``-ed) to tools instead of ``Tainted``.
    cfi_mode:
        ``"control"`` (default): control-flow integrity is lowered only
        by ``if`` conditions. ``"data"``: every argument dependency also
        lowers it (pessimistic). See the module docstring.

    Raises
    ------
    PlanError
        For malformed plans (checked up front via :func:`validate_plan`).
    IFCViolationError
        In strict mode when a step is denied. The partial trace is
        attached as ``exc.trace``.
    PlanToolError
        When a tool itself raises.
    """
    problems = validate_plan(
        plan,
        tools,
        initial_bindings=(initial_bindings or {}).keys(),
        max_steps=max_steps,
    )
    if problems:
        raise PlanError("; ".join(problems))
    if cfi_mode not in ("control", "data"):
        raise ValueError(f"cfi_mode must be 'control' or 'data' (got {cfi_mode!r})")

    ctx = context if context is not None else IFCContext()
    base_cfi: Integrity = ctx.control_flow_integrity
    source_for_tool = dict(source_for_tool or {})

    bindings: dict[str, Any] = {}
    for name, value in (initial_bindings or {}).items():
        bindings[name] = (
            value
            if isinstance(value, Tainted)
            else Tainted(value, provenance=frozenset({f"binding:{name}"}), integrity="TRUSTED")
        )
    trace = PlanTrace(bindings=bindings)

    for index, step in enumerate(plan):
        tool_name: str = step["call"]
        fn = tools[tool_name]
        started = time.perf_counter()
        cfi: Integrity = base_cfi

        # Evaluate the optional condition first; it is a control dependency.
        if "if" in step:
            cond_deps: list[Any] = []
            cond = _eval(step["if"], bindings, cond_deps, depth=0, step=index)
            cond_sources: set[str] = set()
            for dep in (*cond_deps, cond):
                prov, _caps, integrity = labels_of(dep)
                cfi = _meet(cfi, integrity)
                cond_sources |= prov
            for src in sorted(cond_sources):
                ctx.record(src, "if", tool_name)
            if not lower(cond):
                trace.steps.append(
                    StepResult(
                        index=index,
                        tool=tool_name,
                        action="skipped",
                        cfi=cfi,
                        labels=_snapshot(bindings),
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
                continue

        arg_deps: list[Any] = []
        raw_args = step.get("args", {})
        if isinstance(raw_args, dict):
            kwargs = {
                k: _eval(v, bindings, arg_deps, depth=0, step=index) for k, v in raw_args.items()
            }
            args: tuple[Any, ...] = ()
        else:
            kwargs = {}
            args = tuple(_eval(v, bindings, arg_deps, depth=0, step=index) for v in raw_args)

        arg_sources: set[str] = set()
        for dep in arg_deps:
            prov, _caps, integrity = labels_of(dep)
            if cfi_mode == "data":
                cfi = _meet(cfi, integrity)
            arg_sources |= prov
        for src in sorted(arg_sources):
            ctx.record(src, "arg", tool_name)

        step_ctx = IFCContext(
            control_flow_integrity=cfi,
            taint_chain=ctx.taint_chain,
            user_metadata=ctx.user_metadata,
            on_decision=ctx.on_decision,
        )
        metadata: ToolMetadata | None = getattr(fn, "__aisafepy_tool__", None)
        try:
            decision = evaluate_call(
                tool=tool_name,
                args=args,
                kwargs=kwargs,
                policy=policy,
                metadata=metadata,
                context=step_ctx,
                mediator=mediator,
            )
        except IFCViolationError as exc:
            trace.steps.append(
                StepResult(
                    index=index,
                    tool=tool_name,
                    action="block",
                    cfi=cfi,
                    violation=exc.violation,
                    decision=exc.decision,
                    labels=_snapshot(bindings),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            trace.halted_at = index
            trace.violation = exc.violation
            exc.trace = trace  # type: ignore[attr-defined]
            raise

        call_args = tuple(lower(a) for a in args) if unwrap_args else args
        call_kwargs = {k: lower(v) for k, v in kwargs.items()} if unwrap_args else kwargs
        try:
            result = fn(*call_args, **call_kwargs)
        except Exception as exc:
            trace.steps.append(
                StepResult(
                    index=index,
                    tool=tool_name,
                    action="error",
                    cfi=cfi,
                    decision=decision,
                    labels=_snapshot(bindings),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            trace.halted_at = index
            raise PlanToolError(index, tool_name, exc, trace) from exc

        bound_name = step.get("bind")
        if bound_name is not None:
            bindings[bound_name] = policy.lift_output(
                tool_name, result, source=source_for_tool.get(tool_name)
            )
            ctx.record(tool_name, "bind", bound_name)

        trace.steps.append(
            StepResult(
                index=index,
                tool=tool_name,
                action="allow",
                cfi=cfi,
                decision=decision,
                bound=bound_name,
                labels=_snapshot(bindings),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        )

    return trace


def _snapshot(bindings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: _label_dict(v) for name, v in bindings.items()}


# ---- expression evaluation ---------------------------------------------------------


def _eval(expr: Any, bindings: Mapping[str, Any], deps: list[Any], *, depth: int, step: int) -> Any:
    """Evaluate ``expr``; append every referenced binding to ``deps``.

    Returns a ``Tainted`` whenever the result depends on tainted data,
    otherwise the plain literal.
    """
    if depth > MAX_EXPR_DEPTH:
        raise PlanError(f"expression nesting exceeds {MAX_EXPR_DEPTH}", step=step)

    if isinstance(expr, list):
        items = [_eval(e, bindings, deps, depth=depth + 1, step=step) for e in expr]
        return _join_container([lower(i) for i in items], items)

    if not isinstance(expr, dict):
        return expr

    ops = [k for k in expr if isinstance(k, str) and k.startswith("$")]
    if not ops:
        evaluated = {k: _eval(v, bindings, deps, depth=depth + 1, step=step) for k, v in expr.items()}
        return _join_container({k: lower(v) for k, v in evaluated.items()}, evaluated.values())

    op = ops[0]
    body = expr[op]
    if op == "$lit":
        return body
    if op == "$ref":
        if body not in bindings:
            raise PlanError(f"$ref to unbound name {body!r}", step=step)
        value = bindings[body]
        deps.append(value)
        return value
    if op == "$get":
        container = _eval(body[0], bindings, deps, depth=depth + 1, step=step)
        key = _eval(body[1], bindings, deps, depth=depth + 1, step=step)
        raw_key = lower(key)
        try:
            if isinstance(container, Tainted):
                got = container[raw_key]
            else:
                got = container[raw_key]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlanError(f"$get failed: {type(exc).__name__}: {exc}", step=step) from exc
        if isinstance(key, Tainted):
            got = got.join(key) if isinstance(got, Tainted) else key.map(lambda _k: lower(got))
        return got
    if op == "$concat":
        parts = [_eval(e, bindings, deps, depth=depth + 1, step=step) for e in body]
        text = "".join(str(lower(p)) for p in parts)
        return _join_container(text, parts)
    if op == "$format":
        with_ = expr.get("with", {})
        values = {k: _eval(v, bindings, deps, depth=depth + 1, step=step) for k, v in with_.items()}
        try:
            text = body.format(**{k: lower(v) for k, v in values.items()})
        except (KeyError, IndexError, ValueError) as exc:
            raise PlanError(f"$format failed: {type(exc).__name__}: {exc}", step=step) from exc
        return _join_container(text, values.values())
    raise PlanError(f"unknown operator {op!r}", step=step)  # pragma: no cover - validated


def _join_container(value: Any, parts: Iterable[Any]) -> Any:
    """Wrap ``value`` in a ``Tainted`` carrying the joined labels of ``parts``.

    If no part is tainted the plain ``value`` is returned unchanged so
    literal-only expressions stay literal.
    """
    prov: frozenset[str] = frozenset()
    caps: frozenset[str] = frozenset()
    integrity: Integrity = "TRUSTED"
    any_tainted = False
    for p in parts:
        if isinstance(p, Tainted):
            any_tainted = True
            prov = prov | p.provenance
            caps = caps | p.capabilities
            integrity = _meet(integrity, p.integrity)
    if not any_tainted:
        return value
    return Tainted(value, provenance=prov, capabilities=caps, integrity=integrity)
