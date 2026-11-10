"""Allow ``python -m aisafepy`` as an alias for the ``aisafepy`` console script."""

from aisafepy.cli.main import main

raise SystemExit(main())
