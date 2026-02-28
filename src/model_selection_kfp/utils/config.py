from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class SplitConfig:
    method: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int


@dataclass(frozen=True)
class StressLevels:
    noise: List[float]
    missingness: List[float]
    covariate_shift: List[float]


@dataclass(frozen=True)
class SelectionWeights:
    w_quality: float
    w_robustness: float


@dataclass(frozen=True)
class AppConfig:
    dataset_path: str
    target_col: str
    time_col: Optional[str]
    id_cols: List[str]

    split: SplitConfig

    model_candidates: List[str]

    primary_metric: str
    additional_metrics: List[str]

    stress_enabled: bool
    stress_levels: StressLevels

    selection: SelectionWeights

    artifacts_root_dir: str


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: '{key}'")
    return d[key]


def load_config(path: str | Path) -> AppConfig:
    """
    Load YAML config and map into strongly-typed config objects.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    data = _require(raw, "data")
    split = _require(raw, "split")
    models = _require(raw, "models")
    metrics = _require(raw, "metrics")
    stress = _require(raw, "stress_tests")
    selection = _require(raw, "selection")
    artifacts = _require(raw, "artifacts")

    split_cfg = SplitConfig(
        method=str(_require(split, "method")),
        train_ratio=float(_require(split, "train_ratio")),
        val_ratio=float(_require(split, "val_ratio")),
        test_ratio=float(_require(split, "test_ratio")),
        seed=int(_require(split, "seed")),
    )

    # Strictly enforce your standard split
    total = split_cfg.train_ratio + split_cfg.val_ratio + split_cfg.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0. Got {total}")

    # Locked standard (as per your workflow)
    if (round(split_cfg.train_ratio, 2), round(split_cfg.val_ratio, 2), round(split_cfg.test_ratio, 2)) != (0.70, 0.15, 0.15):
        raise ValueError("Split must be 70/15/15 as per project standard.")

    stress_levels = StressLevels(
        noise=list((_require(stress, "noise") or {}).get("levels", [])),
        missingness=list((_require(stress, "missingness") or {}).get("levels", [])),
        covariate_shift=list((_require(stress, "covariate_shift") or {}).get("levels", [])),
    )

    selection_weights = SelectionWeights(
        w_quality=float(_require(selection, "w_quality")),
        w_robustness=float(_require(selection, "w_robustness")),
    )

    wsum = selection_weights.w_quality + selection_weights.w_robustness
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"Selection weights must sum to 1.0. Got {wsum}")

    return AppConfig(
        dataset_path=str(_require(data, "dataset_path")),
        target_col=str(_require(data, "target_col")),
        time_col=data.get("time_col", None),
        id_cols=list(data.get("id_cols", []) or []),

        split=split_cfg,

        model_candidates=list((_require(models, "candidates") or [])),

        primary_metric=str(_require(metrics, "primary")),
        additional_metrics=list(metrics.get("additional", []) or []),

        stress_enabled=bool(_require(stress, "enabled")),
        stress_levels=stress_levels,

        selection=selection_weights,

        artifacts_root_dir=str(_require(artifacts, "root_dir")),
    )