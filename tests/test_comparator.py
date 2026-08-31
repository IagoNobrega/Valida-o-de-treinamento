from core.comparator import compare_models, recommend_training
from core.config import Metrics


def status(old, new, key="map50"):
    return next(row["status"] for row in compare_models(Metrics(**{key: old}), Metrics(**{key: new})) if row["key"] == key)


def test_metric_improved(): assert status(0.8, 0.9) == "MELHOR"
def test_metric_worsened(): assert status(0.9, 0.8) == "PIOR"
def test_metric_stable(): assert status(0.9, 0.9005) == "ESTÁVEL"
def test_lower_is_better(): assert status(10.0, 8.0, "inference_ms") == "MELHOR"


def test_recommend_training_prefers_more_precise_model():
    old = Metrics(precision=0.82, recall=0.80, map50=0.75, map50_95=0.62, inference_ms=22.0)
    new = Metrics(precision=0.87, recall=0.84, map50=0.81, map50_95=0.68, inference_ms=18.0)
    recommendation = recommend_training(old, new)
    assert recommendation["winner"] == "candidato"
    assert "precisão" in recommendation["message"].lower()
    assert "candidato" in recommendation["message"].lower()
