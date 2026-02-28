from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    ingest_dir: Path
    split_dir: Path
    train_dir: Path
    eval_dir: Path
    stress_dir: Path
    score_dir: Path
    report_dir: Path


def build_artifact_paths(root_dir: str) -> ArtifactPaths:
    root = Path(root_dir)

    paths = ArtifactPaths(
        root=root,
        ingest_dir=root / "ingest",
        split_dir=root / "split",
        train_dir=root / "train",
        eval_dir=root / "eval",
        stress_dir=root / "stress",
        score_dir=root / "score",
        report_dir=root / "report",
    )

    for p in [
        paths.root,
        paths.ingest_dir,
        paths.split_dir,
        paths.train_dir,
        paths.eval_dir,
        paths.stress_dir,
        paths.score_dir,
        paths.report_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return paths