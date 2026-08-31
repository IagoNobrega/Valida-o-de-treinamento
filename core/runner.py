from __future__ import annotations

from pathlib import Path
from .config import ClassMetric, Metrics, ValidationConfig
from .metrics import extract_metrics


def run_validation(model_path: Path, config: ValidationConfig, output_dir: Path, role: str, class_names: list[str]) -> tuple[Metrics, list[ClassMetric]]:
    """Run a single validation. Callers must pass the same config instance for both models."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    result = model.val(**config.as_yolo_kwargs(output_dir, role))
    return extract_metrics(result, class_names)
