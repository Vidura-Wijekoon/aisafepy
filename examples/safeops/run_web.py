#!/usr/bin/env python3
"""SafeOps — browser demo.

    python run_web.py            # http://127.0.0.1:8420
    python run_web.py --port 9000
"""

from __future__ import annotations

import argparse

from safeops.web import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8420)
    a = ap.parse_args()
    serve(a.host, a.port)
