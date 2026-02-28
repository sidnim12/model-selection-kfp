from __future__ import annotations

import argparse

from model_selection_kfp.utils.config import load_config
from model_selection_kfp.utils.logging import get_logger
from model_selection_kfp.utils.paths import build_artifact_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the robustness-aware model selection pipeline locally.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    logger = get_logger()
    cfg = load_config(args.config)
    paths = build_artifact_paths(cfg.artifacts_root_dir)

    logger.info("Local run initialized.")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Target: {cfg.target_col}")
    logger.info(f"Split: 70/15/15 | method={cfg.split.method} | seed={cfg.split.seed}")
    logger.info(f"Models: {cfg.model_candidates}")
    logger.info(f"Artifacts root: {paths.root.resolve()}")

    # Phase wiring will be added next (ingest -> split -> train -> eval -> stress -> score -> report)
    logger.info("Next: implement components starting with ingest.py and split.py.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())