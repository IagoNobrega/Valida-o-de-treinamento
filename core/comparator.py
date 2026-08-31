from __future__ import annotations

from dataclasses import asdict
from .config import METRICS_CONFIG, METRIC_TOLERANCE, ClassMetric, Metrics


def metric_status(key: str, old: float | None, new: float | None, tolerance: float = METRIC_TOLERANCE) -> str:
    if old is None or new is None:
        return "INDISPONÍVEL"
    delta = new - old
    if abs(delta) <= tolerance:
        return "ESTÁVEL"
    improved = delta > 0 if METRICS_CONFIG[key]["higher_is_better"] else delta < 0
    return "MELHOR" if improved else "PIOR"


def compare_models(old_metrics: Metrics, new_metrics: Metrics, tolerance: float = METRIC_TOLERANCE) -> list[dict]:
    old, new = asdict(old_metrics), asdict(new_metrics)
    rows = []
    for key, meta in METRICS_CONFIG.items():
        delta = None if old[key] is None or new[key] is None else new[key] - old[key]
        rows.append({"key": key, "metric": meta["label"], "old": old[key], "new": new[key], "delta": delta,
                     "status": metric_status(key, old[key], new[key], tolerance)})
    return rows


def compare_class_metrics(old: list[ClassMetric], new: list[ClassMetric], tolerance: float = METRIC_TOLERANCE) -> list[dict]:
    old_by_name, new_by_name = {item.class_name: item for item in old}, {item.class_name: item for item in new}
    rows = []
    for name in sorted(old_by_name.keys() | new_by_name.keys()):
        before, after = old_by_name.get(name), new_by_name.get(name)
        old_map, new_map = (before.map50 if before else None), (after.map50 if after else None)
        old_recall, new_recall = (before.recall if before else None), (after.recall if after else None)
        rows.append({"class_name": name, "old_map50": old_map, "new_map50": new_map,
                     "delta_map50": None if None in (old_map, new_map) else new_map - old_map,
                     "old_recall": old_recall, "new_recall": new_recall,
                     "delta_recall": None if None in (old_recall, new_recall) else new_recall - old_recall,
                     "status": metric_status("map50", old_map, new_map, tolerance)})
    return rows


def recommend_training(old: Metrics, new: Metrics, class_rows: list[dict] | None = None, tolerance: float = METRIC_TOLERANCE) -> dict:
    primary = [row for row in compare_models(old, new, tolerance) if row["key"] in {"precision", "recall", "map50", "map50_95"}]
    winner = "empate"
    if not primary:
        return {"winner": winner, "message": "Não foi possível definir um vencedor com base nas métricas disponíveis."}

    better = sum(row["status"] == "MELHOR" for row in primary)
    worse = sum(row["status"] == "PIOR" for row in primary)
    stable = sum(row["status"] == "ESTÁVEL" for row in primary)
    class_regression = bool(class_rows and any(row.get("status") == "PIOR" for row in class_rows))

    if better > worse:
        winner = "candidato"
    elif worse > better:
        winner = "referência"

    if winner == "empate":
        message = "Os dois treinamentos estão tecnicamente equivalentes nas métricas principais; escolha pelo histórico ou pela facilidade operacional."
    elif winner == "candidato":
        message = (
            "O treinamento candidato é o melhor para uso com mais precisão. "
            f"Ele apresentou {better} métricas principais melhores, {worse} piores e {stable} estáveis."
        )
    else:
        message = (
            "O treinamento de referência é o mais seguro para uso com mais precisão. "
            f"Ele apresentou {better} métricas principais melhores, {worse} piores e {stable} estáveis."
        )

    if class_regression and winner == "candidato":
        message += " Porém, há regressões por classe, então vale revisar a performance por categoria antes de validar em produção."
    if winner == "candidato" and getattr(new, "map50", None) is not None and getattr(old, "map50", None) is not None:
        message += f" A melhoria principal foi mAP50 de {old.map50:.4f} para {new.map50:.4f}."
    if winner == "referência" and getattr(new, "map50", None) is not None and getattr(old, "map50", None) is not None:
        message += f" A referência manteve melhor mAP50 em {old.map50:.4f} contra {new.map50:.4f}."
    return {"winner": winner, "message": message}


def evaluate_candidate(old: Metrics, new: Metrics, class_rows: list[dict], tolerance: float = METRIC_TOLERANCE) -> str:
    primary = [row for row in compare_models(old, new, tolerance) if row["key"] in {"precision", "recall", "map50", "map50_95"}]
    statuses = [row["status"] for row in primary]
    class_regression = any(row["status"] == "PIOR" for row in class_rows)
    if statuses.count("PIOR") >= 2 or "map50_95" in [row["key"] for row in primary if row["status"] == "PIOR"]:
        return "PIOR"
    if "PIOR" in statuses or class_regression:
        return "ANALISAR"
    return "MELHOR" if "MELHOR" in statuses else "ANALISAR"
