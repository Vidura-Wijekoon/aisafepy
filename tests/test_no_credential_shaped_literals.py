"""Fail if any tracked text file contains a credential-shaped literal.

Test fixtures for the secret detectors must be assembled at runtime
(``"AKIA" + "..."``) rather than written out contiguously. A contiguous
fake key is indistinguishable from a real one to GitHub push protection,
gitleaks and reviewers' scanners, and once committed it lives in history
forever. This test runs the same broad pattern set those tools use over
every tracked text file so the problem is caught before a commit, not
after a rejected push.

Regex *definitions* (``r"\\bsk-ant-[A-Za-z0-9_-]{16,}"``) do not match
because their metacharacters break the token shape; that is intended.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: dict[str, re.Pattern[str]] = {
    "openai/anthropic key": re.compile(
        r"\bsk-(?:ant-)?(?:api\d\d-|proj-|svcacct-|admin-|live-|test-)?[A-Za-z0-9_-]{16,}"
    ),
    "stripe key": re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}"),
    "github token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    "slack webhook": re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
    "aws access key": re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    "google oauth": re.compile(r"\bya29\.[0-9A-Za-z_-]{10,}"),
    "hugging face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "sendgrid key": re.compile(r"\bSG\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "npm token": re.compile(r"\bnpm_[A-Za-z0-9]{20,}"),
    "pypi token": re.compile(r"\bpypi-AgEI[A-Za-z0-9_-]{20,}"),
    "twilio key": re.compile(r"\bSK[0-9a-fA-F]{32}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "pem private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "basic-auth url": re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]{4,}@"),
    "credential assignment": re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{12,}"
    ),
    "azure account key": re.compile(r"AccountKey=[A-Za-z0-9+/=]{20,}"),
}

TEXT_SUFFIXES = {
    ".py", ".md", ".yml", ".yaml", ".toml", ".json", ".jsonl", ".txt", ".cfg",
    ".ini", ".html", ".css", ".js", ".tex", ".bib", ".pyi",
}


def _tracked_text_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available; cannot enumerate tracked files")
    files = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        p = ROOT / raw.decode("utf-8", errors="replace")
        if p.suffix.lower() in TEXT_SUFFIXES and p.is_file():
            files.append(p)
    return files


def test_tracked_files_contain_no_credential_shaped_literals() -> None:
    hits: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            for m in pattern.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                rel = path.relative_to(ROOT).as_posix()
                hits.append(f"{rel}:{line_no}: {name}: {m.group(0)[:60]!r}")
    assert not hits, (
        "credential-shaped literals found; build fixtures at runtime "
        "(e.g. 'AKIA' + '...') instead:\n  " + "\n  ".join(hits)
    )


def test_pattern_set_detects_runtime_assembled_examples() -> None:
    """Sanity check that the patterns above would catch what they claim to."""
    samples = {
        "aws access key": "AKIA" + "IOSFODNN7EXAMPLE",
        "github token": "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789",
        "slack token": "xoxb-" + "123456789012-abcdefghijkl",
        "stripe key": "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc",
        "pem private key": "-----BEGIN RSA PRIVATE" + " KEY-----",
        "basic-auth url": "https://user:" + "p4ssw0rd@host.example",
    }
    for name, sample in samples.items():
        assert PATTERNS[name].search(sample), name
