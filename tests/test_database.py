from pathlib import Path
from database.database import Database


def test_save_comparison(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    payload = {"comparison_id":"CMP-1", "created_at":"2026-01-01T00:00:00", "old_model":str(tmp_path / "old.pt"), "new_model":str(tmp_path / "new.pt"), "old_classes":["a"], "new_classes":["a"], "config":{"data":"data.yaml","split":"test","conf":.25,"iou":.5,"imgsz":640,"device":"cpu"}, "status":"MELHOR", "metrics":{"old":{key:.1 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}, "new":{key:.2 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}}, "class_metrics":{"old":[], "new":[]}}
    db.save_comparison(payload)
    assert db.load_comparison("CMP-1")["comparison"]["result_status"] == "MELHOR"


def test_load_comparison(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    assert db.load_comparison("missing") is None
