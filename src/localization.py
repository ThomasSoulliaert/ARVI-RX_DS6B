"""Localisation FR pour l'AFFICHAGE uniquement (site + PDF).

Principe : la sortie brute du modèle (JSON, base SQLite, évaluation, revue HT)
reste en anglais — c'est la source de vérité, on n'y touche jamais. Cette couche
produit une COPIE d'affichage francisée :

  - vocabulaire contrôlé (classe, qualité, limites connues) → dictionnaire FR
    (fiable, hors-ligne, sans dépendance) ;
  - texte libre (justification, éléments visuels) → traduit UNIQUEMENT si
    `translate_free_text=True` (option C : bouton « Traduire » à la demande),
    via deep-translator, avec repli sur l'anglais si la traduction échoue.

La classe canonique (`predicted_class`) est conservée telle quelle pour ne pas
casser la logique de couleur des badges ; le libellé français est fourni à part
dans `predicted_class_label`.
"""
from __future__ import annotations

from typing import Any

FR_CLASS = {
    "normal": "normal",
    "suspected_opacity": "suspicion d'opacité",
    "uncertain": "incertain",
}
FR_QUALITY = {
    "good": "bonne",
    "medium": "moyenne",
    "poor": "mauvaise",
    "limited": "limitée",
}
# Limites connues (chaînes émises par les garde-fous / le modèle jouet).
FR_LIMITATIONS = {
    "image quality": "qualité image",
    "projection": "projection",
    "lack of clinical context": "absence de contexte clinique",
    "no clinical context": "absence de contexte clinique",
    "synthetic toy image": "image synthétique (jouet)",
    "not a validated medical model": "modèle médical non validé",
    "not a validated model": "modèle non validé",
    "fallback: real vlm unavailable, deterministic toy model used":
        "repli : VLM réel indisponible, modèle jouet déterministe utilisé",
}

_CACHE: dict[tuple[str, str, str], str] = {}


def translate_text(text: Any, target: str = "fr", source: str = "en") -> str:
    """Traduit une chaîne EN→FR (repli : renvoie l'original si indisponible).

    deep-translator est importé paresseusement : sans la lib (ou hors-ligne),
    la fonction renvoie simplement le texte d'origine, sans jamais planter.
    """
    text = str(text)
    if not text.strip():
        return text
    key = (source, target, text)
    if key in _CACHE:
        return _CACHE[key]
    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source=source, target=target).translate(text) or text
    except Exception:
        out = text  # repli silencieux : on garde l'anglais
    _CACHE[key] = out
    return out


def fr_limitation(item: Any, translate: bool) -> str:
    low = str(item).lower().strip()
    if low in FR_LIMITATIONS:
        return FR_LIMITATIONS[low]
    return translate_text(item) if translate else str(item)


def localize_prediction(pred: dict, translate_free_text: bool = False) -> dict:
    """Renvoie une COPIE d'affichage francisée de `pred` (jamais l'original)."""
    view = dict(pred)

    cls = pred.get("predicted_class", "uncertain")
    view["predicted_class"] = cls  # canonique : couleur du badge
    view["predicted_class_label"] = FR_CLASS.get(cls, cls)

    quality = pred.get("image_quality", "")
    view["image_quality_label"] = FR_QUALITY.get(str(quality).lower(), quality)

    justification = pred.get("justification", "")
    view["justification"] = translate_text(justification) if translate_free_text else justification

    visual = pred.get("visual_elements") or pred.get("visual_evidence") or []
    view["visual_elements"] = [
        translate_text(x) if translate_free_text else x for x in visual
    ]
    view["visual_evidence"] = view["visual_elements"]

    view["limitations"] = [
        fr_limitation(item, translate_free_text) for item in pred.get("limitations", [])
    ]
    return view