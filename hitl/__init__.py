"""二期 HITL 对外入口（与 Web / ``api.hitl_*`` / ``run.py hitl`` 共用）。

::

    from hitl import seed, review, resync, bc, compare_three

    seed(from_run=...)   # 整拷一期进 sess
    review(work=...)
    resync(work=...)
    bc(work=...)
    compare_three(work=...)

看板：``python run.py hitl board`` → ``output/hitl_sessions/``
"""
from __future__ import annotations

from common.paths import ensure_repo_path

ensure_repo_path()

from .compare_eval import compare_three
from .metrics_report import compute_absorption, load_effort
from .pipeline import bc, pipeline, resync, review, seed, finalize_review
from .session import SessionPaths, create_session, latest_session, list_sessions

__all__ = [
    "seed",
    "review",
    "resync",
    "bc",
    "finalize_review",
    "pipeline",
    "compare_three",
    "compute_absorption",
    "load_effort",
    "SessionPaths",
    "create_session",
    "latest_session",
    "list_sessions",
]
