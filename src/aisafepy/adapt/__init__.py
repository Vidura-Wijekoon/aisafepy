"""The continuous eval-to-guardrail compiler.

``aisafepy.adapt`` closes the loop between red-team frameworks (PyRIT,
Garak, Inspect AI), production traces (Langfuse, Arize Phoenix), and
the runtime guardrail pipeline. It takes a stream of
``FailureRecord``s, clusters them, and compiles each cluster into one
or more of:

* a distilled classifier (small encoder, drop-in for Tier 2)
* a synthesized regex (Tier 1)
* a Cedar / OPA policy rule
* a steering vector (self-hosted models only)
* a deliberative case (prompt-injected reasoning hint, CADA-style)

The compiled artifacts are bundled in a :class:`CompilationReport`
which can be canary-deployed via :func:`promote` with an explicit
false-positive budget.
"""

from aisafepy.adapt.canary import CanaryResult, promote
from aisafepy.adapt.cluster import Cluster, cluster_failures
from aisafepy.adapt.compile import (
    CompilationReport,
    CompiledArtifact,
    GuardCompiler,
    Target,
)
from aisafepy.adapt.governance import AuditEntry, AuditLog
from aisafepy.adapt.sources import (
    FailureRecord,
    GarakReport,
    InspectLog,
    LangfuseTraces,
    ProductionTraces,
    PyRITSource,
    RedTeamSource,
)

__all__ = [
    "AuditEntry",
    "AuditLog",
    "CanaryResult",
    "Cluster",
    "CompilationReport",
    "CompiledArtifact",
    "FailureRecord",
    "GarakReport",
    "GuardCompiler",
    "InspectLog",
    "LangfuseTraces",
    "ProductionTraces",
    "PyRITSource",
    "RedTeamSource",
    "Target",
    "cluster_failures",
    "promote",
]
