"""Deliberative-case target (CADA, arXiv 2601.08000).

We turn each cluster's prototypical failures into *cases* that the
production system prompt cites at inference time. Case-augmented
reasoning generalizes better than rule-code on open-source models;
this target is the cheapest, model-agnostic way to leverage that.

The output is a markdown fragment ready to be injected into the
system prompt or to be loaded as a RAG document.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from aisafepy.adapt.cluster import Cluster
from aisafepy.adapt.compile import CompiledArtifact


@dataclass
class _DeliberativeTarget:
    policy: str
    """Path to a markdown file with the company's safety policy. Cases
    are appended to / inserted into this file's content."""
    k_shot: int = 8
    kind: str = "deliberative"

    def compile_for_cluster(self, cluster: Cluster) -> CompiledArtifact | None:
        if not cluster.records:
            return None
        cases = [r for r in cluster.records if r.was_violation][: self.k_shot]
        if not cases:
            return None

        policy_text = ""
        policy_path = Path(self.policy) if isinstance(self.policy, (str, Path)) else None
        if policy_path is not None and policy_path.exists():
            policy_text = policy_path.read_text(encoding="utf-8")

        rendered = _render_cases(cluster, cases, policy_text)
        return CompiledArtifact(
            kind=self.kind,
            name=f"deliberative-{cluster.label}",
            payload=rendered,
            cluster_label=cluster.label,
            attack_success_rate=cluster.attack_success_rate,
            n_records=cluster.size,
            summary=cluster.summary,
            metadata={"policy": str(self.policy), "n_cases": len(cases)},
        )


def _render_cases(cluster: Cluster, cases, policy_text: str) -> str:
    body = textwrap.dedent(
        f"""
        # AIsafePy-generated deliberative cases
        # Cluster {cluster.label}. Summary: {cluster.summary}
        # Attack success rate observed: {cluster.attack_success_rate:.2%}

        When you encounter the patterns illustrated below, apply the
        company safety policy verbatim. Treat these as **cases** rather
        than rules: reason about whether the current request matches the
        spirit of one of the cases before responding.
        """
    ).strip()
    for i, c in enumerate(cases, 1):
        body += f"\n\n## Case {i}\n\nInput: {c.input!r}\nReason: this matched cluster {cluster.label}; refuse or transform per policy."
    if policy_text:
        body += "\n\n---\n\n" + policy_text
    return body
