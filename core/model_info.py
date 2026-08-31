from __future__ import annotations

from pathlib import Path
from .config import ModelInfo


def normalize_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def load_model_info(path: Path) -> ModelInfo:
    """Open a model using Ultralytics and return its declared class names."""
    from ultralytics import YOLO

    model = YOLO(str(path))
    return ModelInfo(path=path, names=normalize_names(getattr(model, "names", {})))
