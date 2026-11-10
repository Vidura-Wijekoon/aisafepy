"""Entry point for the ``aisafepy`` command.

The root parser owns three global flags (``--json``, ``--quiet``,
``--version``) and dispatches to one module per command group::

    aisafepy scan       run Tier-1 guards over files or stdin
    aisafepy policy     evaluate a PolicyDocument YAML against an input
    aisafepy audit      verify, tail, append to, and summarise audit logs
    aisafepy manifest   pin and verify MCP tool manifests
    aisafepy compile    compile red-team failures into guard artifacts
    aisafepy decisions  summarise and filter GuardDecision JSONL logs

Every command returns an integer exit code and never lets an exception
escape to the shell: usage problems and unreadable inputs are reported
on one line with exit code 1, and "the tool did its job but found
something" (a blocked file, a broken hash chain, manifest drift) exits
with code 2 so shell scripts and CI steps can branch on it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections.abc import Callable, Sequence
from typing import IO

from aisafepy import __version__
from aisafepy.cli import _io
from aisafepy.cli._io import EXIT_ERROR, EXIT_OK, CLIError

Handler = Callable[[argparse.Namespace, IO[str], IO[str]], int]

PROG = "aisafepy"


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with every command group registered."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Capability-based IFC, cascaded guardrails, and an eval-to-guardrail "
            "compiler for LLM agents. Command-line utilities."
        ),
        epilog="Exit codes: 0 ok, 1 error, 2 flagged (block / verification failure / drift).",
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROG} {__version__}"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of tables where supported",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress informational output; errors and flagged findings still print",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = False

    from aisafepy.cli import (
        audit,
        compile,
        decisions,
        evaluate,
        flow,
        manifest,
        pipeline,
        policy,
        scan,
    )

    for module in (scan, policy, pipeline, flow, evaluate, audit, manifest, compile, decisions):
        module.register(subparsers)

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    stdin: IO[str] | None = None,
) -> int:
    """Run the CLI and return an exit code.

    ``stdout`` / ``stderr`` / ``stdin`` default to the process streams.
    Passing ``io.StringIO`` instances makes the CLI trivially testable
    without subprocesses.
    """
    out = _io.out_stream(stdout)
    err = _io.err_stream(stderr)
    parser = build_parser()

    # argparse writes help / usage errors directly to sys.stdout / sys.stderr
    # and raises SystemExit. Capture both so callers see them on the streams
    # they passed in and never get an exception.
    captured_out, captured_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        _flush(captured_out, out)
        _flush(captured_err, err)
        code = exc.code if isinstance(exc.code, int) else EXIT_ERROR
        return EXIT_OK if code == 0 else EXIT_ERROR

    if args.command is None:
        parser.print_help(file=out)
        return EXIT_OK

    args.stdin = stdin
    handler: Handler = args.handler
    try:
        return int(handler(args, out, err))
    except CLIError as exc:
        _io.emit(err, f"{PROG}: error: {exc}")
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        _io.emit(err, f"{PROG}: interrupted")
        return EXIT_ERROR
    except Exception as exc:  # defensive: never dump a traceback on the user
        _io.emit(err, f"{PROG}: error: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


def _flush(buffer: io.StringIO, target: IO[str]) -> None:
    text = buffer.getvalue()
    if text:
        target.write(text)


def add_common_output_flags(parser: argparse.ArgumentParser, *, formats: Sequence[str]) -> None:
    """Attach a ``--format`` option shared by commands that render tables."""
    parser.add_argument(
        "--format",
        choices=list(formats),
        default=formats[0],
        help=f"output format (default: {formats[0]})",
    )


def wants_json(args: argparse.Namespace) -> bool:
    """True when either the global ``--json`` or a ``--format json`` was given."""
    return bool(getattr(args, "json", False)) or getattr(args, "format", None) == "json"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
