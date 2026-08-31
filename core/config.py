from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

METRIC_TOLERANCE = 0.001

METRICS_CONFIG = {
    "precision": {"label": "Precision", "higher_is_better": True},
    "recall": {"label": "Recall", "higher_is_better": True},
    "map50": {"label": "mAP50", "higher_is_better": True},
    "map50_95": {"label": "mAP50-95", "higher_is_better": True},
    "preprocess_ms": {"label": "Preprocessamento", "higher_is_better": False},
    "inference_ms": {"label": "Inferência", "higher_is_better": False},
    "postprocess_ms": {"label": "Pós-processamento", "higher_is_better": False},
}


@dataclass(frozen=True)
class ValidationConfig:
    """The immutable configuration shared by both model validations."""

    data: Path
    split: Literal["test", "val"] = "test"
    conf: float = 0.25
    iou: float = 0.50
    imgsz: int = 640
    device: str = "cpu"

    def as_yolo_kwargs(self, project: Path, name: str) -> dict:
        return {
            "data": str(self.data), "split": self.split, "conf": self.conf,
            "iou": self.iou, "imgsz": self.imgsz, "device": self.device,
            "project": str(project), "name": name, "exist_ok": True,
            "verbose": False, "plots": False,
        }

    def as_dict(self) -> dict:
        result = asdict(self)
        result["data"] = str(self.data)
        return result


@dataclass
class ModelInfo:
    path: Path
    names: list[str]

    @property
    def class_count(self) -> int:
        return len(self.names)


@dataclass
class Metrics:
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    preprocess_ms: float | None = None
    inference_ms: float | None = None
    postprocess_ms: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass
class ClassMetric:
    class_name: str
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
