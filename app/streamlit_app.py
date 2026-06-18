from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.guardrails import WARNING_TEXT
from src.pipeline import run_prediction


st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")

st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning(WARNING_TEXT)

uploaded = st.file_uploader(
    "Déposer une radiographie thoracique frontale",
    type=["png", "jpg", "jpeg", "bmp"],
)
mode = st.selectbox("Mode", ["baseline", "improved"], index=1)

if uploaded is None:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")
    st.stop()

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
            st.error("L'analyse pédagogique a échoué. Veuillez réessayer avec une image valide.")

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
        st.write(visual_elements if visual_elements else "Aucun élément visuel spécifique.")

        st.write("**Limites**")
        for limitation in prediction.get("limitations", []):
            st.write(f"- {limitation}")

        with st.expander("JSON complet"):
            st.json(prediction)
    else:
        st.info("Cliquer sur le bouton pour lancer l'analyse pédagogique.")
