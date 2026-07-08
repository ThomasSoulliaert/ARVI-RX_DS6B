"""Export PDF d'un résultat d'analyse (image + sortie structurée).

Fonction pure, sans dépendance à Streamlit : elle prend l'image analysée et le
dictionnaire de prédiction (celui produit par `run_prediction` /
`apply_safety_guardrails`) et renvoie les octets d'un PDF. Réutilisable depuis
l'interface Streamlit comme depuis l'API.

Dépendance : fpdf2 (voir requirements.txt). L'avertissement non clinique
obligatoire est toujours inclus dans le PDF.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .guardrails import WARNING_TEXT

# Les polices intégrées de fpdf2 sont en latin-1 : on remplace les caractères
# hors latin-1 (guillemets courbes, tirets longs, puces, œ...) pour éviter tout
# plantage sur une justification renvoyée par le modèle.
_REPLACEMENTS = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "•": "-",
    "→": "->", "œ": "oe", "Œ": "OE", " ": " ",
}


def _latin1(text: Any) -> str:
    text = str(text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _image_as_png(image_source: str | Path | bytes) -> tuple[BytesIO, int, int]:
    """Ouvre l'image (chemin ou octets), la convertit en PNG RGB en mémoire."""
    if isinstance(image_source, (bytes, bytearray)):
        img = Image.open(BytesIO(image_source))
    else:
        img = Image.open(Path(image_source))
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer, img.width, img.height


def build_analysis_pdf(image_source: str | Path | bytes, prediction: dict) -> bytes:
    """Construit un PDF A4 (image + résultat structuré) et renvoie ses octets."""
    from fpdf import FPDF  # import local : la lib n'est requise que pour l'export

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    epw = pdf.epw  # largeur utile (entre marges)

    # --- En-tête ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _latin1("RadioScan IA - Rapport d'analyse"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, _latin1("Prototype pedagogique - resultat experimental, non diagnostique"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # --- Bandeau d'avertissement obligatoire ---
    pdf.set_fill_color(255, 244, 229)
    pdf.set_draw_color(220, 130, 80)
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(epw, 6, _latin1("AVERTISSEMENT : " + prediction.get("warning", WARNING_TEXT)),
                   border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Image + champs clés côte à côte ---
    png, iw, ih = _image_as_png(image_source)
    img_w = 85.0
    img_h = img_w * ih / iw
    top = pdf.get_y()
    pdf.image(png, x=pdf.l_margin, y=top, w=img_w)

    # Colonne de droite : champs clés
    key_x = pdf.l_margin + img_w + 8
    key_w = epw - img_w - 8
    pdf.set_xy(key_x, top)

    def field(label: str, value: Any) -> None:
        pdf.set_x(key_x)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(key_w, 5, _latin1(label), new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(key_x)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(key_w, 6, _latin1(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    conf = prediction.get("confidence", 0.0)
    field("Classe predite", prediction.get("predicted_class_label") or prediction.get("predicted_class", "-"))
    field("Confiance", f"{float(conf):.0%}" if isinstance(conf, (int, float)) else conf)
    field("Qualite image", prediction.get("image_quality_label") or prediction.get("image_quality", "-"))
    field("Mode", f"{prediction.get('mode', '-')} ")
    field("Modele", prediction.get("model_name", "-"))

    # On repart sous le plus bas des deux colonnes.
    pdf.set_y(max(pdf.get_y(), top + img_h) + 4)

    # --- Sections texte ---
    def section(title: str, lines: list[str] | str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 7, _latin1(title), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)
        if isinstance(lines, str):
            lines = [lines]
        if not lines:
            lines = ["-"]
        for line in lines:
            pdf.multi_cell(epw, 5.5, _latin1(f"- {line}" if len(lines) > 1 else line),
                           new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    section("Justification", prediction.get("justification", "-"))
    section("Elements visuels",
            prediction.get("visual_elements") or prediction.get("visual_evidence") or [])
    section("Limites", prediction.get("limitations") or [])

    # --- Pied : métadonnées de traçabilité ---
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    meta = (
        f"case_id : {prediction.get('case_id', '-')}   |   "
        f"latence : {prediction.get('latency_ms', '-')} ms   |   "
        f"genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    pdf.multi_cell(epw, 4.5, _latin1(meta), new_x="LMARGIN", new_y="NEXT")

    out = pdf.output()
    return bytes(out)