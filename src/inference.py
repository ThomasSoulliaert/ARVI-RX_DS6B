from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .preprocessing import basic_quality_flag

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def predict_with_model(image_path: str | Path, mode: str = "improved") -> dict[str, Any]:
    """Model entrypoint used by the web/API pipeline.

    The current implementation delegates to the deterministic toy model. A real
    Hugging Face or local model can replace this function later without changing
    Streamlit, FastAPI, or the central pipeline.
    """
    if mode not in {"baseline", "improved"}:
        raise ValueError("Unsupported mode. Expected 'baseline' or 'improved'.")

    return toy_predict(image_path, mode=mode)


# ---------------------------------------------------------------------------
# Vraie inférence VLM (MedGemma). À compléter par l'étudiant.
# Le modèle est chargé UNE seule fois et mis en cache (le chargement est lent).
# ---------------------------------------------------------------------------
_MODEL = None
_PROCESSOR = None
MODEL_ID = "google/medgemma-4b-it"


def _load_model():
    """Charge le modèle MedGemma et son processor, une seule fois (cache global).

    Indices :
    - importer transformers ICI (pas en haut du fichier) pour que le module
      reste importable sans GPU / sans transformers installé.
    - utiliser AutoModelForImageTextToText.from_pretrained(MODEL_ID, ...)
      et AutoProcessor.from_pretrained(MODEL_ID).
    - sur GPU : device_map="auto" et un dtype adapté (ex. bfloat16).
    """
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        # TODO 1: from transformers import AutoModelForImageTextToText, AutoProcessor
        # TODO 2: _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        # TODO 3: _MODEL = AutoModelForImageTextToText.from_pretrained(MODEL_ID, ...)
        raise NotImplementedError("À compléter : charger MedGemma + processor")
    return _MODEL, _PROCESSOR


def _extract_json(text: str) -> dict[str, Any]:
    """Extrait le bloc JSON de la réponse texte du modèle.

    Le modèle peut écrire du blabla / ```json ... ``` autour du JSON.
    Indices :
    - NE PAS faire json.loads(text) directement.
    - isoler le bloc entre le premier '{' et le dernier '}'.
    - json.loads() dans un try/except : si ça échoue, renvoyer un dict vide {}
      (les garde-fous le rattraperont ensuite en 'uncertain').
    """
    # TODO: localiser et parser le JSON ; retourner {} en cas d'échec.
    raise NotImplementedError("À compléter : parsing JSON robuste")


def vlm_predict(
    image_path: str | Path,
    prompt_path: str | Path = "prompts/baseline_prompt.txt",
    mode: str = "baseline",
) -> dict[str, Any]:
    """Inférence réelle via MedGemma. Même schéma de sortie que toy_predict.

    Cette fonction renvoie le dict BRUT. L'appelant doit ensuite passer le
    résultat dans apply_safety_guardrails() (comme dans api/main.py et eval/).
    """
    start = time.perf_counter()
    model, processor = _load_model()

    # 1. TODO: charger l'image (réutiliser load_image de preprocessing.py)
    # 2. TODO: lire le texte du prompt depuis prompt_path
    # 3. TODO: construire les "messages" (rôle user : IMAGE d'abord, puis TEXTE)
    # 4. TODO: inputs = processor.apply_chat_template(messages, ...) -> tenseurs
    # 5. TODO: outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    #          (do_sample=False = déterministe, indispensable pour comparer les prompts)
    # 6. TODO: décoder UNIQUEMENT les tokens générés (pas le prompt d'entrée) -> texte
    # 7. parsed = _extract_json(texte)

    parsed: dict[str, Any] = {}  # TODO: remplacer par _extract_json(texte)

    latency_ms = int((time.perf_counter() - start) * 1000)

    # 8. Mapper vers le dict-contrat. On garantit les clés "médicales" issues du
    #    modèle (avec valeurs par défaut prudentes si absentes) + la plomberie.
    return {
        "image_quality": parsed.get("image_quality", "limited"),
        "predicted_class": parsed.get("predicted_class", "uncertain"),
        "confidence": parsed.get("confidence", 0.0),
        "visual_evidence": parsed.get("visual_evidence", []),
        "justification": parsed.get("justification", ""),
        "limitations": parsed.get("limitations", ["not a validated medical model"]),
        "warning": WARNING,
        "model_name": MODEL_ID,
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }
