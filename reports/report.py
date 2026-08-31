from __future__ import annotations

from pathlib import Path
from typing import Any


def write_basic_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    lines = [f"# Relatório de validação {payload['comparison_id']}", "", f"**Resultado:** {payload['status']}", ""]
    recommendation = payload.get("recommendation", {}).get("message", "")
    if recommendation:
        lines.append(f"**Recomendação:** {recommendation}")
    lines.extend(["", "## Configuração", ""])
    lines.extend(f"- {key}: {value}" for key, value in payload["config"].items())
    lines.extend(["", "## Métricas", "", "| Métrica | Referência | Candidato | Δ | Status |", "|---|---:|---:|---:|---|"])
    for row in payload["comparison_rows"]:
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        lines.append(f"| {row['metric']} | {fmt(row['old'])} | {fmt(row['new'])} | {fmt(row['delta'])} | {row['status']} |")
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
