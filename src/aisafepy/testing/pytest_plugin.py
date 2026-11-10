"""pytest plugin exposing the testing toolkit as fixtures.

Activate it either through the ``pytest11`` entry point (installed with
the package) or explicitly::

    # conftest.py
    pytest_plugins = ["aisafepy.testing.pytest_plugin"]

Fixtures
--------

``aisafepy_corpus``
    The built-in :data:`aisafepy.testing.corpus.BUILTIN` samples.
``attack_samples`` / ``benign_samples``
    The two label slices of the corpus.
``make_pipeline``
    Factory: ``make_pipeline(tier1=[...], tier2=[...], tier3=[...], **kw)``
    returning a ``GuardPipeline``.
``mutator``
    ``mutator(text, strategies=None, max_variants=None)`` seeded from the
    ``aisafepy_seed`` fixture (override it to change the seed).
``guard``
    :func:`aisafepy.testing.assertions.run_guard`, for one-line decisions.

Marker
------

``@pytest.mark.aisafepy_slow`` marks robustness sweeps so a project can
deselect them with ``-m "not aisafepy_slow"``.

pytest is imported lazily so this module can be imported in
environments where pytest is not installed; the fixtures simply are not
defined there.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised through pytester
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "aisafepy_slow: robustness sweeps over the mutation engine; deselect with -m 'not aisafepy_slow'",
    )


if pytest is not None:

    @pytest.fixture(scope="session")
    def aisafepy_seed() -> int:
        """Override in your own conftest to change the mutation seed."""
        return 0

    @pytest.fixture(scope="session")
    def aisafepy_corpus() -> list[Any]:
        from aisafepy.testing.corpus import samples

        return samples()

    @pytest.fixture(scope="session")
    def attack_samples() -> list[Any]:
        from aisafepy.testing.corpus import attacks

        return attacks()

    @pytest.fixture(scope="session")
    def benign_samples() -> list[Any]:
        from aisafepy.testing.corpus import benign

        return benign()

    @pytest.fixture
    def make_pipeline() -> Any:
        from aisafepy.stream.pipeline import GuardPipeline

        def factory(
            tier1: list[Any] | None = None,
            tier2: list[Any] | None = None,
            tier3: list[Any] | None = None,
            **kw: Any,
        ) -> GuardPipeline:
            return GuardPipeline(
                tier1=list(tier1 or []), tier2=list(tier2 or []), tier3=list(tier3 or []), **kw
            )

        return factory

    @pytest.fixture
    def mutator(aisafepy_seed: int) -> Any:
        from aisafepy.testing.mutations import DEFAULT_STRATEGIES, mutate

        def run(text: str, strategies: Any = None, max_variants: int | None = None) -> list[Any]:
            return mutate(
                text,
                strategies=DEFAULT_STRATEGIES if strategies is None else strategies,
                seed=aisafepy_seed,
                max_variants=max_variants,
            )

        return run

    @pytest.fixture
    def guard() -> Any:
        from aisafepy.testing.assertions import run_guard

        return run_guard
