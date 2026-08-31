from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.comparator import compare_class_metrics, compare_models, evaluate_candidate, recommend_training
from core.config import METRICS_CONFIG, ClassMetric, Metrics, ValidationConfig
from core.runner import run_validation
from core.training_artifacts import TrainingArtifact, inspect_training_folder
from core.validator import ModelCompatibilityError, ValidationError, available_devices, prepare_comparison
from database.database import Database
from reports.report import write_basic_report

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
REFERENCE_LABEL = "Referência"
CANDIDATE_LABEL = "Candidato"
REFERENCE_NAME = "referência"
CANDIDATE_NAME = "candidato"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOGS / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
DB = Database(ROOT / "database" / "analytics.db")


def comparison_id() -> str:
    prefix = f"CMP-{datetime.now():%Y%m%d}-"
    existing = {item["id"] for item in DB.list_comparisons()}
    sequence = 1
    while f"{prefix}{sequence:04d}" in existing or (RESULTS / f"{prefix}{sequence:04d}").exists():
        sequence += 1
    return f"{prefix}{sequence:04d}"


def read_learning_curve(artifact: TrainingArtifact) -> pd.DataFrame:
    if artifact.results_csv is None:
        raise ValidationError("A pasta não contém results.csv, necessário para analisar as curvas de aprendizado.")
    try:
        curve = pd.read_csv(artifact.results_csv)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise ValidationError(f"Não foi possível ler results.csv: {exc}") from exc
    curve.columns = [str(column).strip() for column in curve.columns]
    if curve.empty:
        raise ValidationError("O results.csv não contém épocas de treinamento.")
    if "epoch" not in curve.columns:
        curve.insert(0, "epoch", range(1, len(curve) + 1))
    return curve


def metric_columns(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    common = set(old.select_dtypes(include="number").columns) & set(new.select_dtypes(include="number").columns)
    return sorted(column for column in common if column != "epoch")


def metric_winner(old_value: float, new_value: float, metric_name: str) -> str:
    is_lower_better = any(token in metric_name.lower() for token in ("loss", "time", "dfl", "cls", "box"))
    if abs(new_value - old_value) <= 1e-9:
        return "empate"
    return CANDIDATE_NAME if (new_value < old_value if is_lower_better else new_value > old_value) else REFERENCE_NAME


def summarize_wins(selected: list[str], old_curve: pd.DataFrame, new_curve: pd.DataFrame) -> dict[str, int]:
    wins = {REFERENCE_NAME: 0, CANDIDATE_NAME: 0}
    for metric in selected:
        old_value = float(old_curve[metric].iloc[-1])
        new_value = float(new_curve[metric].iloc[-1])
        winner = metric_winner(old_value, new_value, metric)
        if winner != "empate":
            wins[winner] += 1
    return wins


def render_learning_curves(old_run: TrainingArtifact, new_run: TrainingArtifact) -> None:
    old_curve, new_curve = read_learning_curve(old_run), read_learning_curve(new_run)
    available = metric_columns(old_curve, new_curve)
    if not available:
        st.warning("Não existem métricas numéricas em comum nos dois arquivos results.csv.")
        return
    preferred = [column for column in available if any(token in column.lower() for token in ("map", "precision", "recall", "loss", "fitness"))]
    selected = st.multiselect("Curvas exibidas", available, default=preferred or available[:1], key="curve_metrics")
    if not selected:
        return

    wins = summarize_wins(selected, old_curve, new_curve)

    if wins[CANDIDATE_NAME] > wins[REFERENCE_NAME]:
        recommendation = "O treinamento candidato está melhor de acordo com as métricas finais e pode ser escolhido como opção mais precisa."
        reason = "Motivo: melhor desempenho nas métricas principais e menor custo em métricas de erro."
        st.success("Escolha: treinamento candidato")
        st.caption(reason)
    elif wins[REFERENCE_NAME] > wins[CANDIDATE_NAME]:
        recommendation = "O treinamento de referência está melhor de acordo com as métricas finais e é a opção mais segura para uso."
        reason = "Motivo: melhor estabilidade e desempenho nas métricas principais para uso em produção."
        st.info("Escolha: treinamento de referência")
        st.caption(reason)
    else:
        recommendation = "As duas curvas ficaram equivalentes na última época; a decisão deve considerar também a validação no mesmo dataset."
        reason = "Motivo: os ganhos ficaram equilibrados entre referência e candidato."
        st.warning("Escolha: empate entre os treinamentos")
        st.caption(reason)

    pie = go.Figure(data=[go.Pie(labels=[REFERENCE_LABEL, CANDIDATE_LABEL], values=[wins[REFERENCE_NAME], wins[CANDIDATE_NAME]], hole=0.35, marker_colors=["#5DADE2", "#58D68D"])])
    pie.update_layout(title="Vitórias por métrica selecionada", showlegend=True)
    st.plotly_chart(pie, use_container_width=True)

    comparison = []
    for metric in selected:
        old_value = float(old_curve[metric].iloc[-1])
        new_value = float(new_curve[metric].iloc[-1])
        comparison.append({"Métrica": metric, REFERENCE_LABEL: old_value, CANDIDATE_LABEL: new_value})
    bar = go.Figure()
    bar.add_trace(go.Bar(x=[row["Métrica"] for row in comparison], y=[row[REFERENCE_LABEL] for row in comparison], name=REFERENCE_LABEL, marker_color="#5DADE2"))
    bar.add_trace(go.Bar(x=[row["Métrica"] for row in comparison], y=[row[CANDIDATE_LABEL] for row in comparison], name=CANDIDATE_LABEL, marker_color="#58D68D"))
    bar.update_layout(title="Comparação final das métricas selecionadas", xaxis_title="Métrica", yaxis_title="Valor", barmode="group")
    st.plotly_chart(bar, use_container_width=True)

    st.caption(recommendation)

    chart = go.Figure()
    for metric in selected:
        chart.add_trace(go.Scatter(x=old_curve["epoch"], y=old_curve[metric], mode="lines", name=f"{REFERENCE_LABEL} — {metric}"))
        chart.add_trace(go.Scatter(x=new_curve["epoch"], y=new_curve[metric], mode="lines", name=f"{CANDIDATE_LABEL} — {metric}", line={"dash": "dash"}))
    chart.update_layout(title="Curvas de aprendizado por época", xaxis_title="Época", yaxis_title="Valor", hovermode="x unified")
    st.plotly_chart(chart, width="stretch")

    rows = []
    for metric in selected:
        old_value, new_value = float(old_curve[metric].iloc[-1]), float(new_curve[metric].iloc[-1])
        rows.append({"Métrica": metric, f"{REFERENCE_LABEL} (última época)": old_value, f"{CANDIDATE_LABEL} (última época)": new_value, "Δ": new_value - old_value})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Esta comparação descreve o histórico de treinamento. A aprovação técnica do candidato exige a validação dos dois modelos no mesmo dataset.")


def render_training_info(title: str, artifact: TrainingArtifact) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.caption(f"Pasta: {artifact.root}")
        st.write(f"Pesos: `{artifact.best_weights}`")
        st.write(f"results.csv: `{artifact.results_csv}`" if artifact.results_csv else "results.csv não encontrado")
        if artifact.final_training_metrics:
            st.json(artifact.final_training_metrics)


def curve_analysis_page() -> None:
    st.subheader("Comparação por curvas de aprendizado")
    st.info("Informe as duas pastas de treinamento já existentes neste computador/servidor. Não envie as pastas pelo navegador.")
    with st.form("learning_curves"):
        old_folder = st.text_input("Pasta do treinamento de referência", placeholder=r"C:\caminho\treino_antigo")
        new_folder = st.text_input("Pasta do treinamento candidato", placeholder=r"C:\caminho\treino_novo")
        submitted = st.form_submit_button("Comparar curvas", type="primary")
    if not submitted:
        return
    try:
        old_run, new_run = inspect_training_folder(Path(old_folder)), inspect_training_folder(Path(new_folder))
        first, second = st.columns(2)
        with first:
            render_training_info("Treinamento de referência", old_run)
        with second:
            render_training_info("Treinamento candidato", new_run)
        render_learning_curves(old_run, new_run)
    except ValidationError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Erro ao analisar curvas")
        st.error("Não foi possível analisar as curvas. Consulte logs/app.log.")


def persist_result(cid: str, prepared, old_run: TrainingArtifact, new_run: TrainingArtifact, old_metrics: Metrics, new_metrics: Metrics, old_classes: list[ClassMetric], new_classes: list[ClassMetric], rows: list[dict], class_rows: list[dict], status: str, recommendation: dict) -> None:
    directory = RESULTS / cid
    (directory / "old").mkdir(parents=True, exist_ok=True)
    (directory / "new").mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison_id": cid, "created_at": datetime.now().isoformat(timespec="seconds"),
        "old_model": str(prepared.old_model.path), "new_model": str(prepared.new_model.path),
        "old_classes": prepared.old_model.names, "new_classes": prepared.new_model.names,
        "dataset_classes": prepared.dataset_classes, "config": prepared.config.as_dict(), "status": status,
        "recommendation": recommendation,
        "training_runs": {"old": old_run.as_dict(), "new": new_run.as_dict()},
        "metrics": {"old": old_metrics.as_dict(), "new": new_metrics.as_dict()},
        "class_metrics": {"old": [item.__dict__ for item in old_classes], "new": [item.__dict__ for item in new_classes]},
        "comparison_rows": rows, "class_comparison_rows": class_rows,
    }
    (directory / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(directory / "class_metrics.csv", index=False)
    write_basic_report(directory, payload)
    DB.save_comparison(payload)


def render_metrics(old: Metrics, new: Metrics, rows: list[dict], class_rows: list[dict], status: str, recommendation: dict) -> None:
    st.success(f"Resultado técnico: {status} — {recommendation['message']}")
    cards = st.columns(2)
    for container, title, metrics in [(cards[0], "Referência", old), (cards[1], "Candidato", new)]:
        with container:
            st.subheader(title)
            for key in ("precision", "recall", "map50", "map50_95"):
                value = getattr(metrics, key)
                st.metric(METRICS_CONFIG[key]["label"], "—" if value is None else f"{value * 100:.2f}%")
    table = pd.DataFrame(rows)
    if not table.empty:
        st.dataframe(table[["metric", "old", "new", "delta", "status"]], hide_index=True, width="stretch")
    if class_rows:
        st.subheader("Métricas por classe")
        st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")


def validation_page() -> None:
    st.subheader("Validação técnica no mesmo dataset")
    st.caption("Usa automaticamente weights/best.pt de cada pasta e aplica a mesma configuração aos dois modelos.")
    with st.form("validation"):
        old_folder = st.text_input("Pasta de referência", key="validation_old")
        new_folder = st.text_input("Pasta candidata", key="validation_new")
        data_yaml = st.text_input("data.yaml de validação")
        first, second, third = st.columns(3)
        split = first.selectbox("Split", ["test", "val"])
        conf = second.number_input("Confidence", 0.0, 1.0, 0.25, 0.01)
        iou = third.number_input("IoU", 0.0, 1.0, 0.50, 0.01)
        imgsz = first.number_input("Image size", 32, 4096, 640, 32)
        device = second.selectbox("Device", available_devices(), format_func=lambda value: "CPU" if value == "cpu" else f"GPU {value}")
        submitted = st.form_submit_button("Executar validação", type="primary")
    if not submitted:
        return
    cid = comparison_id()
    try:
        old_run, new_run = inspect_training_folder(Path(old_folder)), inspect_training_folder(Path(new_folder))
        config = ValidationConfig(Path(data_yaml), split, float(conf), float(iou), int(imgsz), device)
        with st.status("Validando os dois modelos...", expanded=True) as progress:
            prepared = prepare_comparison(old_run.best_weights, new_run.best_weights, config)
            progress.write("Executando referência...")
            old_metrics, old_classes = run_validation(prepared.old_model.path, prepared.config, RESULTS / cid, "old", prepared.dataset_classes)
            progress.write("Executando candidato...")
            new_metrics, new_classes = run_validation(prepared.new_model.path, prepared.config, RESULTS / cid, "new", prepared.dataset_classes)
            rows, class_rows = compare_models(old_metrics, new_metrics), compare_class_metrics(old_classes, new_classes)
            status = evaluate_candidate(old_metrics, new_metrics, class_rows)
            recommendation = recommend_training(old_metrics, new_metrics, class_rows)
            persist_result(cid, prepared, old_run, new_run, old_metrics, new_metrics, old_classes, new_classes, rows, class_rows, status, recommendation)
            progress.update(label="Validação concluída", state="complete")
        render_metrics(old_metrics, new_metrics, rows, class_rows, status, recommendation)
    except ModelCompatibilityError as exc:
        st.error("INCOMPATIBILIDADE ENTRE MODELOS")
        st.json({"referência": exc.old_classes, "candidato": exc.new_classes, "dataset": exc.dataset_classes})
    except ValidationError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Erro na validação técnica")
        st.error("Ocorreu um erro inesperado. Consulte logs/app.log.")


def history_page() -> None:
    records = DB.list_comparisons()
    if not records:
        st.info("Ainda não há comparações técnicas salvas.")
        return
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")


st.set_page_config(page_title="YOLO Training Analytics", layout="wide")
st.title("YOLO Training Analytics")
st.caption("Comparação e análise de desempenho entre versões de modelos YOLO.")
curves_tab, validation_tab, history_tab = st.tabs(["Curvas de aprendizado", "Validação técnica", "Histórico"])
with curves_tab:
    curve_analysis_page()
with validation_tab:
    validation_page()
with history_tab:
    history_page()
