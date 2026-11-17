"""``python -m hitl`` → 转调 ``run.py hitl`` 同源逻辑（见 ``hitl.cli``）。"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
