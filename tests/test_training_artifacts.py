from core.training_artifacts import inspect_training_folder, shared_dataset_path


def make_run(root, class_name="item"):
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "best.pt").touch()
    (root / "args.yaml").write_text("epochs: 20\nimgsz: 640\n", encoding="utf-8")
    (root / "results.csv").write_text("epoch,metrics/mAP50(B)\n19,0.91\n", encoding="utf-8")


def test_inspect_training_folder_finds_standard_yolo_files(tmp_path):
    make_run(tmp_path / "run")
    artifact = inspect_training_folder(tmp_path / "run")
    assert artifact.best_weights.name == "best.pt"
    assert artifact.args_yaml.name == "args.yaml"
    assert artifact.final_training_metrics["metrics/mAP50(B)"] == 0.91


def test_shared_dataset_requires_identical_yaml(tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    make_run(old_root); make_run(new_root)
    (old_root / "data.yaml").write_text("names: [item]", encoding="utf-8")
    (new_root / "data.yaml").write_text("names: [item]", encoding="utf-8")
    assert shared_dataset_path(inspect_training_folder(old_root), inspect_training_folder(new_root)) == old_root / "data.yaml"
