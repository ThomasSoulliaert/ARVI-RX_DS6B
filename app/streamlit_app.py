from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image

from src.database import fetch_recent_runs, summarize_runs
from src.guardrails import WARNING_TEXT
from src.pipeline import run_prediction

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")

def render_analysis_page() -> None:
    uploaded = st.file_uploader(
        "Déposer une radiographie thoracique frontale",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    mode = st.selectbox("Mode", ["baseline", "improved"], index=1)

    if uploaded is None:
        st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")
        return

    suffix = Path(uploaded.name).suffix or ".png"
    file_bytes = uploaded.getvalue()
    upload_key = f"{uploaded.name}:{len(file_bytes)}"

    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("last_prediction", None)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    col_image, col_result = st.columns([1, 1])

    with col_image:
        st.image(
            Image.open(tmp_path),
            caption="Image uploadée",
            use_container_width=True,
        )

    with col_result:
        if st.button("Analyser l'image", type="primary"):
            try:
                result = run_prediction(str(tmp_path), mode=mode, save=True)
                st.session_state["last_prediction"] = result
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                st.error(
                    "L'analyse pédagogique a échoué. Veuillez réessayer avec une image valide."
                )

        prediction = st.session_state.get("last_prediction")
        if prediction:
            st.subheader("Résultat expérimental")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Classe prédite", prediction["predicted_class"])
            metric_cols[1].metric("Confiance", prediction["confidence"])
            metric_cols[2].metric("Qualité image", prediction["image_quality"])

            st.write("**Justification**")
            st.write(prediction["justification"])

            st.write("**Éléments visuels**")
            visual_elements = prediction.get("visual_elements", [])
            st.write(
                visual_elements if visual_elements else "Aucun élément visuel spécifique."
            )

            st.write("**Limites**")
            for limitation in prediction.get("limitations", []):
                st.write(f"- {limitation}")

            with st.expander("JSON complet"):
                st.json(prediction)
        else:
            st.info("Cliquer sur le bouton pour lancer l'analyse pédagogique.")

def _dashboard_rows(recent_runs: list[dict]) -> list[dict]:
    rows = []
    for run in recent_runs:
        prediction = run.get("prediction", {})
        rows.append(
            {
                "id": run.get("id"),
                "date": run.get("created_at"),
                "case_id": run.get("case_id"),
                "fichier": Path(run.get("image_path") or "").name,
                "mode": run.get("mode") or prediction.get("mode", ""),
                "classe": run.get("predicted_class"),
                "confiance": run.get("confidence"),
                "qualité": run.get("image_quality") or prediction.get("image_quality", ""),
                "latence_ms": run.get("latency_ms"),
            }
        )
    return rows

def render_dashboard_page() -> None:
    summary = summarize_runs()
    recent_runs = fetch_recent_runs(limit=20)

    if summary["total"] == 0:
        st.info("Aucune prédiction sauvegardée pour le moment.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Prédictions", summary["total"])
    metric_cols[1].metric("Confiance moyenne", summary["average_confidence"])
    metric_cols[2].metric("Latence moyenne", summary["average_latency_ms"])
    metric_cols[3].metric("Classes distinctes", len(summary["class_counts"]))

    st.subheader("Répartition des classes")
    for predicted_class, count in summary["class_counts"].items():
        st.write(f"- {predicted_class}: {count}")

    st.subheader("Dernières prédictions")
    st.dataframe(_dashboard_rows(recent_runs), use_container_width=True, hide_index=True)

    st.subheader("Sorties JSON récentes")
    for run in recent_runs[:5]:
        label = f"Run {run.get('id')} - {run.get('predicted_class')} - {run.get('created_at')}"
        with st.expander(label):
            st.json(run.get("prediction", {}))

st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning(WARNING_TEXT)

page = st.sidebar.radio(
    "Navigation",
    ["Analyse image", "Historique / Dashboard"],
)

if page == "Analyse image":
    render_analysis_page()
else:
    render_dashboard_page()