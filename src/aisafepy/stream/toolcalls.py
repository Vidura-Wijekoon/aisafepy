"""Tier-1 policy checks at the tool-call boundary.

``aisafepy.flow`` answers *where did this argument come from?* This
module answers the complementary, label-free question *is this
argument acceptable on its face?*: is the tool allowed at all, are the
arguments the ones the schema declares, does a URL point at a private
network, does a path climb out of its sandbox, does a shell command
smuggle a second command behind a semicolon.

The checks are organised as three reusable policy objects
(:class:`URLPolicy`, :class:`PathPolicy`, :class:`ShellPolicy`) so other
guards (for example :class:`aisafepy.stream.heuristics.URLGuard`) can
apply the same rules to free text, and one :class:`ToolCallGuard` that
walks a tool call's arguments and applies them.

Tool-call shapes understood by :func:`extract_tool_calls`:

* ``{"name": ..., "arguments": {...} | "<json string>"}``
* OpenAI: ``{"type": "function", "function": {"name", "arguments"}}``
  and ``{"tool_calls": [...]}`` envelopes
* Anthropic: ``{"type": "tool_use", "name": ..., "input": {...}}``
* loose: ``{"tool": ..., "args" | "parameters" | "input": {...}}``
* a JSON list of any of the above
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit

from aisafepy.core.decisions import GuardDecision
from aisafepy.stream.pipeline import Context

# --- tool call extraction ---------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool invocation."""

    name: str
    arguments: dict[str, Any]
    raw: Any = field(compare=False, hash=False, default=None)


def _coerce_args(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    if isinstance(value, dict):
        return value
    return {"_value": value}


def _iter_calls(obj: Any) -> Iterator[ToolCall]:
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_calls(item)
        return
    if not isinstance(obj, dict):
        return
    if isinstance(obj.get("tool_calls"), list):
        yield from _iter_calls(obj["tool_calls"])
        return
    fn = obj.get("function")
    if isinstance(fn, dict) and "name" in fn:
        yield ToolCall(str(fn["name"]), _coerce_args(fn.get("arguments")), obj)
        return
    if obj.get("type") == "tool_use" and "name" in obj:
        yield ToolCall(str(obj["name"]), _coerce_args(obj.get("input")), obj)
        return
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if name is None:
        return
    for key in ("arguments", "args", "parameters", "params", "input", "inputs"):
        if key in obj:
            yield ToolCall(str(name), _coerce_args(obj[key]), obj)
            return
    yield ToolCall(str(name), {}, obj)


def extract_tool_calls(obj_or_text: Any) -> list[ToolCall]:
    """Parse tool calls out of a JSON string, dict or list.

    Raises ``ValueError`` when given a string that is not JSON. Returns
    an empty list for JSON that contains no recognisable tool call.
    """
    obj = obj_or_text
    if isinstance(obj, str):
        stripped = obj.strip()
        if not stripped:
            return []
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("not a JSON tool call") from exc
    return list(_iter_calls(obj))


# --- URL policy -------------------------------------------------------------

_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa")
_LOCAL_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal", "instance-data"})


def _host_as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse dotted, bracketed, decimal (``2130706433``), hex and octal IP forms."""
    h = host.strip("[]")
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        pass
    parts = h.split(".")
    if not 1 <= len(parts) <= 4 or not all(re.fullmatch(r"0x[0-9a-fA-F]+|\d+", p) for p in parts):
        return None
    try:
        values = [_parse_c_int(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 1:
        n = values[0]
        return ipaddress.IPv4Address(n) if 0 <= n <= 0xFFFFFFFF else None
    if len(values) == 4 and all(0 <= v <= 255 for v in values):
        return ipaddress.IPv4Address(".".join(str(v) for v in values))
    return None


def _parse_c_int(token: str) -> int:
    """Parse an integer the way inet_aton does: ``0x`` hex, leading-zero octal, else decimal."""
    if token.lower().startswith("0x"):
        return int(token, 16)
    if len(token) > 1 and token.startswith("0"):
        return int(token, 8)
    return int(token, 10)


def domain_matches(host: str, pattern: str) -> bool:
    """``host`` equals ``pattern`` or is a subdomain of it (case-insensitive)."""
    host = host.lower().rstrip(".")
    pattern = pattern.lower().lstrip("*.").rstrip(".")
    return host == pattern or host.endswith("." + pattern)


@dataclass
class URLPolicy:
    """Rules for a single URL. :meth:`check` returns rule names that fired."""

    allowed_domains: tuple[str, ...] = ()
    """If non-empty, the host must match one of these (subdomains allowed)."""
    denied_domains: tuple[str, ...] = ()
    allowed_schemes: tuple[str, ...] = ("http", "https")
    block_ip_literals: bool = True
    block_private_networks: bool = True
    block_userinfo: bool = True
    block_punycode: bool = False
    max_length: int = 2048

    def check(self, url: str) -> list[str]:
        violations: list[str] = []
        if len(url) > self.max_length:
            violations.append("url_too_long")
        if not _URL_SCHEME_RE.match(url):
            return [*violations, "url_missing_scheme"]
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in self.allowed_schemes:
            violations.append(f"url_scheme_not_allowed:{scheme}")
        netloc = parts.netloc
        if "@" in netloc or "\\" in netloc:
            if self.block_userinfo:
                violations.append("url_userinfo")
        host = (parts.hostname or "").rstrip(".").lower()
        if not host:
            return [*violations, "url_missing_host"]

        if any(ord(c) > 127 for c in host):
            try:
                host_ascii = host.encode("idna").decode("ascii")
            except UnicodeError:
                return [*violations, "url_invalid_idn_host"]
            violations.append("url_idn_host")
            host = host_ascii
        if self.block_punycode and any(label.startswith("xn--") for label in host.split(".")):
            violations.append("url_punycode_host")

        ip = _host_as_ip(host)
        if ip is not None:
            if self.block_ip_literals:
                violations.append("url_ip_literal")
            if self.block_private_networks and (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                violations.append("url_private_network")
        elif self.block_private_networks and (
            host in _LOCAL_HOSTS or host.endswith(_LOCAL_HOST_SUFFIXES) or "." not in host
        ):
            violations.append("url_private_network")

        if self.denied_domains and any(domain_matches(host, d) for d in self.denied_domains):
            violations.append("url_domain_denied")
        if self.allowed_domains and not any(domain_matches(host, d) for d in self.allowed_domains):
            violations.append("url_domain_not_allowed")
        return violations


# --- path policy ------------------------------------------------------------

_SENSITIVE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:^|/)etc/(?:passwd|shadow|sudoers|hosts|ssh/)",
        r"(?:^|/)proc/(?:self|\d+)/",
        r"(?:^|/)\.ssh(?:/|$)",
        r"(?:^|/)id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?$",
        r"(?:^|/)\.env(?:\.[\w-]+)?$",
        r"(?:^|/)\.git/(?:config|credentials)",
        r"(?:^|/)\.(?:aws|kube|docker|gnupg|config/gh)/",
        r"(?:^|/)\.(?:netrc|npmrc|pypirc|bash_history|zsh_history|git-credentials)$",
        r"(?:^|/)windows/system32/config/(?:sam|system|security)",
        r"(?:^|/)windows/(?:ntds|repair)/",
        r"(?:^|/)(?:boot\.ini|ntuser\.dat)$",
        r"(?:^|/)var/run/docker\.sock$",
    )
)


def _to_posix(path: str) -> str:
    """Normalise separators and percent-encoding so checks see one form."""
    p = unquote(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", p):
        p = "/" + p[0].lower() + p[2:]
    return p


@dataclass
class PathPolicy:
    """Rules for a filesystem path argument."""

    allowed_roots: tuple[str, ...] = ()
    """If non-empty, absolute paths must live under one of these roots."""
    deny_traversal: bool = True
    deny_sensitive: bool = True
    extra_sensitive: tuple[str, ...] = ()
    deny_home_expansion: bool = True

    def check(self, path: str) -> list[str]:
        violations: list[str] = []
        posix = _to_posix(path)
        segments = [s for s in posix.split("/") if s]
        if self.deny_traversal and ".." in segments:
            violations.append("path_traversal")
        if self.deny_home_expansion and (posix.startswith("~") or "/~" in posix):
            violations.append("path_home_expansion")
        if self.deny_sensitive:
            if any(p.search(posix) for p in _SENSITIVE_PATH_PATTERNS):
                violations.append("path_sensitive")
            for extra in self.extra_sensitive:
                if re.search(extra, posix, re.IGNORECASE):
                    violations.append("path_sensitive")
                    break
        if self.allowed_roots and _is_absolute(path):
            norm = PurePosixPath(posix)
            if not any(_under(norm, _to_posix(root)) for root in self.allowed_roots):
                violations.append("path_outside_allowed_roots")
        return violations


def _is_absolute(path: str) -> bool:
    return PurePosixPath(path.replace("\\", "/")).is_absolute() or PureWindowsPath(path).is_absolute() or path.startswith("~")


def _under(path: PurePosixPath, root: str) -> bool:
    try:
        path.relative_to(PurePosixPath(root))
        return True
    except ValueError:
        return False


# --- shell policy -----------------------------------------------------------

_SHELL_META_RE = re.compile(r"[;&|`$><\n\r]|\$\(|\{\s*:\s*;?\s*\}")
_DANGEROUS_COMMAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(p, re.IGNORECASE))
    for name, p in (
        ("rm_recursive", r"\brm\s+(?:-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r|--recursive)\b"),
        ("disk_destroy", r"\b(?:mkfs(?:\.\w+)?\b|dd\s+if=|shred\b|wipefs\b)"),
        ("fork_bomb", r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),
        ("pipe_to_shell", r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b"),
        ("reverse_shell", r"\b(?:nc|ncat|netcat)\b[^;&|]*\s-e\s|/dev/tcp/"),
        ("encoded_powershell", r"\bpowershell(?:\.exe)?\b[^\n]*\s-(?:enc|encodedcommand|e)\b"),
        ("decode_and_run", r"\bbase64\s+(?:-d|--decode)\b[^|]*\|\s*(?:ba|z)?sh\b"),
        ("world_writable", r"\bchmod\s+(?:-R\s+)?[0-7]*777\b"),
        ("credential_dump", r"\bcat\s+[^\n;&|]*(?:/etc/shadow|\.ssh/id_|\.aws/credentials)"),
    )
)


@dataclass
class ShellPolicy:
    """Rules for command strings handed to shell-capable tools."""

    shell_tools: tuple[str, ...] = (
        "bash", "sh", "shell", "exec", "execute", "run_command", "run_shell", "terminal",
        "command", "subprocess", "powershell", "cmd",
    )
    """Tool names whose string arguments are treated as commands."""
    command_arg_names: tuple[str, ...] = ("command", "cmd", "script", "shell_command", "args")
    """Argument names treated as commands for *any* tool."""
    deny_metacharacters: bool = True
    deny_dangerous_commands: bool = True

    def is_command_context(self, tool: str, arg_name: str) -> bool:
        tool_l = tool.lower()
        leaf = arg_name.rsplit(".", 1)[-1].lower()
        return tool_l in self.shell_tools or any(tool_l.endswith("_" + t) for t in self.shell_tools) or leaf in self.command_arg_names

    def check(self, command: str) -> list[str]:
        violations: list[str] = []
        if self.deny_metacharacters and _SHELL_META_RE.search(command):
            violations.append("shell_metacharacters")
        if self.deny_dangerous_commands:
            violations.extend(f"shell_dangerous:{name}" for name, rx in _DANGEROUS_COMMAND_PATTERNS if rx.search(command))
        return violations


# --- guard ------------------------------------------------------------------

_PATH_ARG_RE = re.compile(r"(?:^|_)(?:path|file|filename|filepath|dir|directory|folder|dest|destination|src|source|target|location|output|input)s?$", re.IGNORECASE)
_PATH_VALUE_RE = re.compile(r"^(?:~|\.{1,2}[\\/]|[\\/]|[A-Za-z]:[\\/]|\\\\)")
_URL_ARG_RE = re.compile(r"(?:^|_)(?:url|uri|href|link|endpoint|webhook|callback)s?$", re.IGNORECASE)


@dataclass(frozen=True)
class ToolCallViolation:
    tool: str
    arg: str
    rule: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"tool": self.tool, "arg": self.arg, "rule": self.rule, "detail": self.detail}


@dataclass
class ToolCallGuard:
    """Validate tool calls against allow-lists and argument policies.

    Parameters
    ----------
    allowed_tools:
        If given, any tool not in the set is a violation.
    denied_tools:
        Always a violation.
    arg_allowlist:
        Per-tool set of permitted top-level argument names. Unknown
        arguments (a classic way to smuggle extra parameters) are
        violations. Tools absent from the map are not checked.
    url_policy / path_policy / shell_policy:
        Pass ``None`` to disable a family. URL checks run on any string
        argument that carries a scheme or whose name looks URL-ish;
        path checks on arguments whose name or value looks path-ish;
        shell checks on command contexts (see
        :meth:`ShellPolicy.is_command_context`).
    max_arg_chars / max_total_chars:
        Size caps on individual string arguments and on the serialized
        call.
    severity_overrides:
        Map from rule prefix to severity; default ``critical`` for shell
        and private-network rules, ``high`` otherwise.
    """

    name: str = "tool-calls"
    tier: int = 1
    allowed_tools: frozenset[str] | None = None
    denied_tools: frozenset[str] = frozenset()
    arg_allowlist: dict[str, frozenset[str]] | None = None
    url_policy: URLPolicy | None = field(default_factory=URLPolicy)
    path_policy: PathPolicy | None = field(default_factory=PathPolicy)
    shell_policy: ShellPolicy | None = field(default_factory=ShellPolicy)
    max_arg_chars: int = 10_000
    max_total_chars: int = 100_000
    max_depth: int = 16

    def __post_init__(self) -> None:
        if self.allowed_tools is not None:
            self.allowed_tools = frozenset(self.allowed_tools)
        self.denied_tools = frozenset(self.denied_tools)
        if self.arg_allowlist is not None:
            self.arg_allowlist = {k: frozenset(v) for k, v in self.arg_allowlist.items()}

    # ---- public API ----------------------------------------------------

    def check_call(self, call: ToolCall) -> list[ToolCallViolation]:
        """Return every violation for one tool call."""
        v: list[ToolCallViolation] = []
        tool = call.name
        if tool in self.denied_tools:
            v.append(ToolCallViolation(tool, "", "tool_denied"))
        if self.allowed_tools is not None and tool not in self.allowed_tools:
            v.append(ToolCallViolation(tool, "", "tool_not_allowed"))
        if self.arg_allowlist is not None and tool in self.arg_allowlist:
            for extra in sorted(set(call.arguments) - self.arg_allowlist[tool]):
                v.append(ToolCallViolation(tool, extra, "arg_not_allowed"))
        try:
            serialized = json.dumps(call.arguments, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            serialized = str(call.arguments)
        if len(serialized) > self.max_total_chars:
            v.append(ToolCallViolation(tool, "", "args_too_large", f"{len(serialized)} chars"))
        for arg_path, value in _walk(call.arguments, max_depth=self.max_depth):
            if arg_path == "__depth__":
                v.append(ToolCallViolation(tool, "", "args_too_deep"))
                continue
            if not isinstance(value, str):
                continue
            if len(value) > self.max_arg_chars:
                v.append(ToolCallViolation(tool, arg_path, "arg_too_large", f"{len(value)} chars"))
            v.extend(self._check_string_arg(tool, arg_path, value))
        return v

    def check_calls(self, calls: Iterable[ToolCall]) -> list[ToolCallViolation]:
        out: list[ToolCallViolation] = []
        for c in calls:
            out.extend(self.check_call(c))
        return out

    async def __call__(self, ctx: Context) -> GuardDecision:
        start = time.perf_counter()
        calls, parse_evidence = self._calls_from_context(ctx)
        if calls is None:
            return GuardDecision.allow(
                tier=self.tier,
                rationale="content is not a tool call",
                guard_name=self.name,
                evidence=parse_evidence,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        try:
            violations = self.check_calls(calls)
        except Exception as exc:  # pragma: no cover - defensive
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{self.name}: checker error; failing open",
                guard_name=self.name,
                evidence={"error": repr(exc)},
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not violations:
            return GuardDecision.allow(
                tier=self.tier,
                rationale=f"{len(calls)} tool call(s) within policy",
                guard_name=self.name,
                evidence={"tool_calls": [c.name for c in calls]},
                latency_ms=latency_ms,
            )
        rules = sorted({x.rule for x in violations})
        critical = any(r.startswith(("shell_", "url_private_network", "path_sensitive", "tool_denied")) for r in rules)
        summary = "; ".join(f"{x.tool}.{x.arg or '*'}: {x.rule}" for x in violations[:3])
        return GuardDecision.block(
            tier=self.tier,
            confidence=1.0,
            rationale=f"tool call policy violation: {summary}",
            fallback="The requested tool call was blocked by policy.",
            severity="critical" if critical else "high",
            guard_name=self.name,
            evidence={
                "violations": [x.to_dict() for x in violations],
                "rules": rules,
                "tool_calls": [c.name for c in calls],
            },
            latency_ms=latency_ms,
        )

    # ---- internals -----------------------------------------------------

    def _calls_from_context(self, ctx: Context) -> tuple[list[ToolCall] | None, dict[str, Any]]:
        meta = ctx.metadata or {}
        for key in ("tool_call", "tool_calls"):
            if key in meta:
                return extract_tool_calls(meta[key]), {"source": f"metadata.{key}"}
        text = ctx.buffer or ctx.chunk or ""
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return None, {"non_json": True}
        try:
            calls = extract_tool_calls(stripped)
        except ValueError:
            return None, {"non_json": True}
        if not calls:
            return None, {"tool_calls": 0}
        return calls, {"source": "content"}

    def _check_string_arg(self, tool: str, arg: str, value: str) -> list[ToolCallViolation]:
        out: list[ToolCallViolation] = []
        leaf = arg.rsplit(".", 1)[-1]
        stripped = value.strip()
        if self.url_policy is not None and (_URL_SCHEME_RE.match(stripped) or _URL_ARG_RE.search(leaf)):
            for rule in self.url_policy.check(stripped):
                out.append(ToolCallViolation(tool, arg, rule, _redact_url(stripped)))
        if self.path_policy is not None and (_PATH_ARG_RE.search(leaf) or _PATH_VALUE_RE.match(stripped)) and not _URL_SCHEME_RE.match(stripped):
            for rule in self.path_policy.check(stripped):
                out.append(ToolCallViolation(tool, arg, rule, stripped[:120]))
        if self.shell_policy is not None and self.shell_policy.is_command_context(tool, arg):
            for rule in self.shell_policy.check(value):
                out.append(ToolCallViolation(tool, arg, rule, value[:120]))
        return out


def _walk(value: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 16) -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted.path, leaf)`` pairs; emits ``("__depth__", None)`` when too deep."""
    if depth > max_depth:
        yield "__depth__", None
        return
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, prefix=f"{prefix}.{k}" if prefix else str(k), depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk(v, prefix=f"{prefix}[{i}]", depth=depth + 1, max_depth=max_depth)
    else:
        yield prefix, value


def _redact_url(url: str) -> str:
    """Strip userinfo and query from a URL before it lands in evidence."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}{parts.path[:60]}"


__all__ = [
    "PathPolicy",
    "ShellPolicy",
    "ToolCall",
    "ToolCallGuard",
    "ToolCallViolation",
    "URLPolicy",
    "domain_matches",
    "extract_tool_calls",
]
