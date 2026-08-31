from pathlib import Path
from database.database import Database


def test_save_comparison(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    payload = {"comparison_id":"CMP-1", "created_at":"2026-01-01T00:00:00", "old_model":str(tmp_path / "old.pt"), "new_model":str(tmp_path / "new.pt"), "old_classes":["a"], "new_classes":["a"], "config":{"data":"data.yaml","split":"test","conf":.25,"iou":.5,"imgsz":640,"device":"cpu"}, "status":"MELHOR", "product_name":"Produto A", "metrics":{"old":{key:.1 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}, "new":{key:.2 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}}, "class_metrics":{"old":[], "new":[]}}
    db.save_comparison(payload)
    assert db.load_comparison("CMP-1")["comparison"]["result_status"] == "MELHOR"
    assert db.list_comparisons()[0]["product_name"] == "Produto A"


def test_load_comparison(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    assert db.load_comparison("missing") is None


def test_products_are_registered(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    payload = {"comparison_id":"CMP-2", "created_at":"2026-01-02T00:00:00", "old_model":str(tmp_path / "old2.pt"), "new_model":str(tmp_path / "new2.pt"), "old_classes":["a"], "new_classes":["a"], "config":{"data":"data.yaml","split":"test","conf":.25,"iou":.5,"imgsz":640,"device":"cpu"}, "status":"REFERÊNCIA", "product_name":"Produto B", "metrics":{"old":{key:.1 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}, "new":{key:.2 for key in ("precision","recall","map50","map50_95","preprocess_ms","inference_ms","postprocess_ms")}}, "class_metrics":{"old":[], "new":[]}}
    db.save_comparison(payload)
    products = db.list_products()
    assert any(product["product_name"] == "Produto B" for product in products)
