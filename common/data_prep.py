"""训练/评估工作目录：按 scenario 路径拷表到 ROOT/output/..."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .paths import ROOT, mkdir
from .scenario import Scenario, get_scenario, validate_scenario_data


def _src(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else ROOT / p


def prepare_train_workdir(
    scenario: Scenario | None = None,
    *,
    work: Path | None = None,
    chdir: bool = True,
    validate: bool = True,
) -> Path:
    sc = scenario or get_scenario()
    if validate:
        validate_scenario_data(sc, need_train=True, need_eval=False)
    work = Path(mkdir(work if work else ROOT / "output" / "_scratch" / "train_workdir"))
    for dest, rel in (
        ("forecast.csv", sc.forecast_rel),
        ("actual.csv", sc.actual_rel),
        ("growth_para.csv", sc.growth_para_rel),
    ):
        shutil.copy2(_src(rel), work / dest)
    if chdir:
        os.chdir(work)
    return work


def prepare_eval_workdir(
    work: Path | None = None,
    scenario: Scenario | None = None,
    *,
    validate: bool = True,
) -> Path:
    sc = scenario or get_scenario()
    if validate:
        validate_scenario_data(sc, need_train=False, need_eval=True)
    work = Path(mkdir(work if work else ROOT / "output" / "_scratch" / "eval_workdir"))
    for dest, rel in (
        ("forecast.csv", sc.forecast_eval_rel),
        ("actual.csv", sc.actual_eval_rel),
        ("growth_para.csv", sc.growth_para_rel),
    ):
        shutil.copy2(_src(rel), work / dest)
    return work
