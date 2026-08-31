from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from .config import ModelInfo, ValidationConfig
from .model_info import load_model_info, normalize_names


class ValidationError(ValueError):
    """An error that can be presented safely to an application user."""


class ModelCompatibilityError(ValidationError):
    def __init__(self, old: list[str], new: list[str], dataset: list[str]) -> None:
        super().__init__("INCOMPATIBILIDADE ENTRE MODELOS")
        self.old_classes, self.new_classes, self.dataset_classes = old, new, dataset


@dataclass
class PreparedComparison:
    old_model: ModelInfo
    new_model: ModelInfo
    dataset_classes: list[str]
    config: ValidationConfig


def _resolve_dataset_entry(entry: Any, yaml_path: Path) -> list[Path]:
    base = yaml_path.parent
    if isinstance(entry, (list, tuple)):
        return [Path(str(item)) if Path(str(item)).is_absolute() else base / str(item) for item in entry]
    if not isinstance(entry, str):
        return []
    candidate = Path(entry)
    return [candidate if candidate.is_absolute() else base / candidate]


def read_dataset_yaml(path: Path, split: str) -> tuple[dict[str, Any], list[str], list[Path]]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"Não foi possível ler data.yaml: {exc}") from exc
    if not isinstance(content, dict):
        raise ValidationError("O data.yaml deve conter um objeto YAML válido.")
    if split not in content:
        raise ValidationError(f"O split '{split}' não existe no data.yaml.")
    names = normalize_names(content.get("names", {}))
    if not names:
        raise ValidationError("O data.yaml não contém classes válidas em 'names'.")
    root = Path(str(content.get("path", "")))
    if root and not root.is_absolute():
        root = path.parent / root
    raw_paths = _resolve_dataset_entry(content[split], path)
    resolved = [item if item.is_absolute() else root / item for item in raw_paths] if root else raw_paths
    if not resolved or not any(item.exists() for item in resolved):
        raise ValidationError(f"Não foi encontrado diretório/arquivo de imagens para o split '{split}'.")
    return content, names, resolved


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices.extend(str(index) for index in range(torch.cuda.device_count()))
    except ImportError:
        pass
    return devices


def prepare_comparison(old_path: Path, new_path: Path, config: ValidationConfig) -> PreparedComparison:
    old_path, new_path, yaml_path = old_path.resolve(), new_path.resolve(), config.data.resolve()
    if not old_path.is_file() or old_path.suffix.lower() != ".pt":
        raise ValidationError("O modelo de referência .pt não existe ou é inválido.")
    if not new_path.is_file() or new_path.suffix.lower() != ".pt":
        raise ValidationError("O modelo candidato .pt não existe ou é inválido.")
    if old_path == new_path:
        raise ValidationError("Os modelos de referência e candidato devem ser arquivos diferentes.")
    if not yaml_path.is_file():
        raise ValidationError("O arquivo data.yaml não existe.")
    if config.device not in available_devices():
        raise ValidationError(f"O dispositivo solicitado ({config.device}) não está disponível.")
    _, dataset_classes, _ = read_dataset_yaml(yaml_path, config.split)
    try:
        old_info, new_info = load_model_info(old_path), load_model_info(new_path)
    except Exception as exc:
        raise ValidationError(f"Não foi possível abrir um dos modelos YOLO: {exc}") from exc
    if old_info.names != new_info.names or old_info.names != dataset_classes:
        raise ModelCompatibilityError(old_info.names, new_info.names, dataset_classes)
    return PreparedComparison(old_info, new_info, dataset_classes, config)
