from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database import DEFAULT_DB_PATH, fetch_recent_runs, insert_run, summarize_runs
from src.guardrails import WARNING_TEXT
from src.pipeline import run_prediction


def _call_remote_api(api_url: str, file_name: str, file_bytes: bytes, mode: str) -> dict:
    """Appelle POST /predict d'une API distante (ex. FastAPI sur Colab GPU)."""
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        files={"file": (file_name, file_bytes)},
        data={"mode": mode},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _remote_model_state(api_url: str) -> str:
    """Warmup automatique du modèle distant : 'ready', 'loading' ou 'unreachable'.

    Appelle POST /warmup (idempotent, non bloquant côté serveur) jusqu'à ce que
    le modèle soit chargé. Les timeouts pendant le chargement sont volontairement
    silencieux : seul un petit message d'attente est montré à l'utilisateur.
    """
    key = f"remote_ready::{api_url}"
    if st.session_state.get(key):
        return "ready"
    try:
        response = requests.post(f"{api_url.rstrip('/')}/warmup", timeout=10)
    except requests.exceptions.Timeout:
        return "loading"  # le serveur est occupé (chargement) : on repollera
    except requests.RequestException:
        return "unreachable"
    if response.status_code == 404:
        # API d'une version antérieure sans /warmup : ne pas bloquer l'analyse.
        st.session_state[key] = True
        return "ready"
    try:
        status = response.json()
    except ValueError:
        return "loading"
    if status.get("loaded") or status.get("error"):
        # Modèle prêt — ou chargement impossible (ex. token manquant) : dans ce
        # cas /predict utilisera le repli jouet, inutile de bloquer le bouton.
        st.session_state[key] = True
        return "ready"
    return "loading"

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")


def render_remote_sidebar() -> tuple[str, str | None]:
    """Champ URL API distante, rendu sur TOUTES les pages.

    Le widget doit être rendu à chaque exécution du script (avec une key
    stable), sinon Streamlit efface sa valeur dès qu'on visite une page qui ne
    l'affiche pas. Rendu global = l'URL collée survit à la navigation.
    """
    api_url = st.sidebar.text_input(
        "URL API distante (optionnel)",
        value=os.environ.get("ARVI_API_URL", ""),
        key="remote_api_url",
        help=(
            "URL d'une API FastAPI du projet tournant sur une machine GPU "
            "(ex. tunnel Colab https://xxxx.trycloudflare.com). "
            "Laisser vide pour exécuter le modèle localement."
        ),
    ).strip()

    remote_state = None
    if api_url:
        remote_state = _remote_model_state(api_url)
        if remote_state == "ready":
            st.sidebar.success("Modèle distant prêt — inférence déportée sur l'API.")
        elif remote_state == "loading":
            st.sidebar.info("⏳ Chargement du modèle sur le serveur distant…")
        else:
            st.sidebar.warning("API distante injoignable — vérifier l'URL du tunnel.")
    # Tant que le modèle distant charge, on repolle silencieusement (voir le
    # bloc d'auto-rafraîchissement en bas de script), même depuis le dashboard.
    st.session_state["_remote_loading"] = remote_state == "loading"
    return api_url, remote_state


def render_analysis_page(api_url: str, remote_state: str | None) -> None:
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

    st.divider()
    col_image, col_result = st.columns(2, gap="large")

    with col_image:
        st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
        analyze_disabled = bool(api_url) and remote_state != "ready"
        analyze_clicked = st.button(
            "Analyser l'image",
            type="primary",
            use_container_width=True,
            disabled=analyze_disabled,
        )
        if analyze_disabled and remote_state == "loading":
            st.caption("⏳ Le modèle charge sur le serveur distant — le bouton s'activera automatiquement.")

    with col_result:
        if analyze_clicked:
            with st.spinner("Analyse en cours…"):
                try:
                    if api_url:
                        result = _call_remote_api(api_url, uploaded.name, file_bytes, mode)
                        result["remote_api"] = api_url
                        # Log local du run distant pour que le dashboard le voie aussi.
                        try:
                            insert_run(
                                DEFAULT_DB_PATH,
                                result.get("case_id", Path(uploaded.name).stem),
                                f"remote:{uploaded.name}",
                                result,
                            )
                            result["saved"] = True
                        except Exception:
                            result["saved"] = False
                    else:
                        result = run_prediction(str(tmp_path), mode=mode, save=True)
                    st.session_state["last_prediction"] = result
                except requests.RequestException as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 524 or isinstance(exc, requests.exceptions.Timeout):
                        # 524 = timeout du tunnel Cloudflare (~100 s) : typiquement le
                        # premier appel d'une session, pendant que le serveur GPU
                        # télécharge et charge encore les poids du modèle.
                        st.warning(
                            "Le serveur distant n'a pas répondu à temps : le modèle est "
                            "probablement en cours de chargement sur le GPU (premier appel "
                            "de la session, ~2 à 5 min). Réessayer dans une minute — les "
                            "appels suivants prendront ~15-20 s."
                        )
                    else:
                        st.error(
                            f"Appel de l'API distante échoué ({exc}). "
                            "Vérifier que l'API tourne et que l'URL du tunnel est à jour."
                        )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error(
                        "L'analyse pédagogique a échoué. Veuillez réessayer avec une image valide."
                    )

        prediction = st.session_state.get("last_prediction")
        if prediction:
            if prediction.get("is_toy"):
                st.warning(
                    "⚠️ **Modèle jouet utilisé.** Le modèle réel (MedGemma) n'était pas "
                    "disponible sur cette machine (GPU ou token Hugging Face manquant). "
                    "Ce résultat est produit par une règle déterministe qui valide "
                    "uniquement la chaîne logicielle : il ne reflète aucune analyse "
                    "réelle de l'image."
                )
            st.subheader("Résultat expérimental")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Classe prédite", prediction["predicted_class"])
            metric_cols[1].metric("Confiance", prediction["confidence"])
            metric_cols[2].metric("Qualité image", prediction["image_quality"])

            st.divider()

            with st.container(border=True):
                st.markdown("**Justification**")
                st.write(prediction["justification"])

            with st.container(border=True):
                st.markdown("**Éléments visuels**")
                visual_elements = prediction.get("visual_elements", [])
                if visual_elements:
                    for element in visual_elements:
                        st.write(f"- {element}")
                else:
                    st.write("Aucun élément visuel spécifique.")

            with st.container(border=True):
                st.markdown("**Limites**")
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
                "qualité": run.get("image_quality")
                or prediction.get("image_quality", ""),
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

    st.divider()
    col_chart, col_list = st.columns([1.3, 1])

    with col_chart:
        st.subheader("Répartition des classes")
        if summary["class_counts"]:
            st.bar_chart(summary["class_counts"])
        else:
            st.write("Aucune donnée.")

    with col_list:
        st.subheader("Détail")
        for predicted_class, count in summary["class_counts"].items():
            st.write(f"- {predicted_class} : {count}")

    st.divider()
    st.subheader("Dernières prédictions")
    st.dataframe(_dashboard_rows(recent_runs), use_container_width=True, hide_index=True)

    st.subheader("Sorties JSON récentes")
    for run in recent_runs[:5]:
        label = f"Run {run.get('id')} - {run.get('predicted_class')} - {run.get('created_at')}"
        with st.expander(label):
            st.json(run.get("prediction", {}))


EVAL_RESULTS_DIR = ROOT / "eval" / "results"


def render_eval_results_page() -> None:
    """Page 'Résultats d'évaluation' : lit les fichiers commités de eval/results/
    (métriques, matrices, figures, registre d'erreurs) pour rendre la démo
    autonome, sans dépendre de la base SQLite locale ni du GPU distant."""
    import pandas as pd

    if not EVAL_RESULTS_DIR.exists():
        st.info("Aucun dossier eval/results/ trouvé dans le dépôt.")
        return

    run_dirs = sorted([d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        st.info("Aucun run d'évaluation commité dans eval/results/.")
        return

    run_dir = Path(
        st.selectbox("Run d'évaluation", run_dirs, format_func=lambda p: p.name)
    )
    st.caption(
        f"Source : `{run_dir.relative_to(ROOT)}` — fichiers produits par "
        "`eval/run_evaluation.py`, figures par `eval/make_figures.py`. "
        "Aucune valeur retapée à la main."
    )

    # --- Synthèse avant/après -------------------------------------------------
    summary_path = run_dir / "before_after_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.subheader("Comparaison baseline / improved")
        if {"mode", "accuracy", "sensitivity_opacity", "uncertain_rate"} <= set(summary.columns):
            cols = st.columns(len(summary))
            for col, (_, row) in zip(cols, summary.iterrows()):
                with col:
                    st.markdown(f"**{row['mode']}** (n = {int(row['n'])})")
                    st.metric("Accuracy", f"{row['accuracy']:.2%}")
                    st.metric("Sensibilité opacités", f"{row['sensitivity_opacity']:.2%}")
                    st.metric("Taux d'incertitude", f"{row['uncertain_rate']:.2%}")
                    st.metric("Latence médiane", f"{row['latency_median_ms'] / 1000:.1f} s")
        with st.expander("Tableau complet des métriques"):
            st.dataframe(summary, use_container_width=True, hide_index=True)

    # --- Figures ---------------------------------------------------------------
    figures_dir = run_dir / "figures"
    figure_paths = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    if figure_paths:
        st.subheader("Figures")
        for fig_path in figure_paths:
            st.image(str(fig_path), use_container_width=True)

    # --- Matrices de confusion (CSV commités) ----------------------------------
    confusion_paths = sorted(run_dir.glob("*_confusion.csv"))
    if confusion_paths:
        st.subheader("Matrices de confusion (données brutes)")
        conf_cols = st.columns(len(confusion_paths))
        for col, conf_path in zip(conf_cols, confusion_paths):
            with col:
                st.markdown(f"`{conf_path.name}`")
                st.dataframe(pd.read_csv(conf_path, index_col=0), use_container_width=True)

    # --- Registre d'erreurs -----------------------------------------------------
    register_path = run_dir / "error_register.csv"
    if register_path.exists():
        st.subheader("Registre d'erreurs (cas commentés)")
        register = pd.read_csv(register_path)
        type_filter = st.multiselect(
            "Filtrer par type d'erreur",
            sorted(register["error_type"].unique()),
            default=sorted(register["error_type"].unique()),
        )
        filtered = register[register["error_type"].isin(type_filter)]
        st.caption(
            f"{len(filtered)} cas affichés / {len(register)} — "
            "FN faux négatif · FP faux positif · UA incertitude · "
            "CM erreur de comportement modèle · OK cas de référence"
        )
        st.dataframe(filtered, use_container_width=True, hide_index=True)


st.title("Assistant radiologue virtuel")
st.caption("Prototype pédagogique d'aide à l'analyse de radiographies thoraciques")
st.warning(WARNING_TEXT)

page = st.sidebar.radio(
    "Navigation", ["Analyse image", "Historique / Dashboard", "Résultats d'évaluation"]
)
api_url, remote_state = render_remote_sidebar()

if page == "Analyse image":
    render_analysis_page(api_url, remote_state)
elif page == "Historique / Dashboard":
    render_dashboard_page()
else:
    render_eval_results_page()

# Auto-rafraîchissement discret : tant que le modèle distant charge, on re-teste
# /warmup toutes les 3 s sans afficher d'erreur, jusqu'à activation du bouton.
if st.session_state.get("_remote_loading"):
    time.sleep(3)
    st.rerun()
