"""Command-line interface for AIsafePy.

Installed as the ``aisafepy`` console script. See :mod:`aisafepy.cli.main`
for the command tree and :doc:`docs/cli.md` for the reference manual.
The CLI depends only on the core package: it never imports torch,
transformers, or an agent framework.
"""

from aisafepy.cli.main import main

__all__ = ["main"]
