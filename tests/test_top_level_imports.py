"""Smoke tests: ensure the package and submodules import cleanly without
any optional heavy dependencies."""

from __future__ import annotations


def test_top_level_imports():
    import aisafepy

    assert aisafepy.__version__
    assert hasattr(aisafepy, "GuardDecision")
    assert hasattr(aisafepy, "Budget")
    assert hasattr(aisafepy, "ProgressTracker")


def test_lazy_submodule_import_flow():
    import aisafepy

    flow = aisafepy.flow
    assert hasattr(flow, "Policy")
    assert hasattr(flow, "Capability")
    assert hasattr(flow, "Tainted")


def test_lazy_submodule_import_stream():
    import aisafepy

    stream = aisafepy.stream
    assert hasattr(stream, "GuardPipeline")
    assert hasattr(stream, "RegexGuard")


def test_lazy_submodule_import_adapt():
    import aisafepy

    adapt = aisafepy.adapt
    assert hasattr(adapt, "GuardCompiler")
    assert hasattr(adapt, "Target")


def test_lazy_submodule_import_contrib():
    import aisafepy

    contrib = aisafepy.contrib
    assert hasattr(contrib, "PromptGuard2")
    assert hasattr(contrib, "LlamaGuard4")
