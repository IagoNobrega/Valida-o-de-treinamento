from __future__ import annotations

from typing import Any
import numpy as np

from .config import ClassMetric, Metrics


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(result: Any, class_names: list[str]) -> tuple[Metrics, list[ClassMetric]]:
    """Map public Ultralytics validation results without fabricating absent fields."""
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", {}) or {}
    metrics = Metrics(
        precision=_number(getattr(box, "mp", None)), recall=_number(getattr(box, "mr", None)),
        map50=_number(getattr(box, "map50", None)), map50_95=_number(getattr(box, "map", None)),
        preprocess_ms=_number(speed.get("preprocess")), inference_ms=_number(speed.get("inference")),
        postprocess_ms=_number(speed.get("postprocess")),
    )
    curves = getattr(box, "p", None), getattr(box, "r", None), getattr(box, "ap50", None), getattr(box, "ap", None)
    if not all(value is not None for value in curves):
        return metrics, []
    arrays = [np.asarray(value).reshape(-1) for value in curves]
    count = min(len(class_names), *(len(value) for value in arrays))
    return metrics, [ClassMetric(class_names[i], *(_number(values[i]) for values in arrays)) for i in range(count)]
