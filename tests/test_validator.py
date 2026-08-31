from pathlib import Path
import pytest
from core.config import ValidationConfig
from core.config import ModelInfo
from core.validator import ModelCompatibilityError, ValidationError, prepare_comparison


def test_reject_same_model(tmp_path):
    model = tmp_path / "model.pt"; model.touch()
    with pytest.raises(ValidationError, match="diferentes"):
        prepare_comparison(model, model, ValidationConfig(tmp_path / "data.yaml"))


def test_same_validation_config_used_for_both_models():
    config = ValidationConfig(Path("data.yaml"), "test", .25, .5, 640, "cpu")
    assert config.as_yolo_kwargs(Path("results"), "old")["data"] == config.as_yolo_kwargs(Path("results"), "new")["data"]
    assert config.as_yolo_kwargs(Path("results"), "old")["imgsz"] == config.as_yolo_kwargs(Path("results"), "new")["imgsz"]


def test_detect_incompatible_classes(tmp_path, monkeypatch):
    old, new, data = tmp_path / "old.pt", tmp_path / "new.pt", tmp_path / "data.yaml"
    old.touch(); new.touch(); data.write_text("names: [a]")
    monkeypatch.setattr("core.validator.available_devices", lambda: ["cpu"])
    monkeypatch.setattr("core.validator.read_dataset_yaml", lambda *args: ({}, ["a"], [tmp_path]))
    monkeypatch.setattr("core.validator.load_model_info", lambda path: ModelInfo(path, ["a"] if path.name == "old.pt" else ["b"]))
    with pytest.raises(ModelCompatibilityError):
        prepare_comparison(old, new, ValidationConfig(data))
