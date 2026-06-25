from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {
    "image_quality",
    "predicted_class",
    "confidence",
    "justification",
    "limitations",
    "warning",
    "model_name",
    "prompt_version",
    "latency_ms",
}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."
ALLOWED_QUALITIES = {"good", "medium", "poor"}
DEFAULT_LIMITATIONS = [
    "Résultat expérimental non clinique.",
    "Validation humaine par un professionnel qualifié requise.",
]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [str(value)]


def normalize_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(pred)

    image_quality = normalized.get("image_quality", "poor")
    if image_quality == "limited":
        image_quality = "poor"
    if image_quality not in ALLOWED_QUALITIES:
        image_quality = "poor"
    normalized["image_quality"] = image_quality

    if normalized.get("predicted_class") not in ALLOWED_CLASSES:
        normalized["predicted_class"] = "uncertain"

    try:
        confidence = float(normalized.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    normalized["confidence"] = round(max(0.0, min(confidence, 1.0)), 3)

    visual_elements = normalized.get("visual_elements", normalized.get("visual_evidence", []))
    normalized["visual_elements"] = _as_list(visual_elements)
    normalized["visual_evidence"] = normalized["visual_elements"]

    justification = str(normalized.get("justification", "")).strip()
    normalized["justification"] = justification or "Sortie prudente du prototype pédagogique."

    limitations = _as_list(normalized.get("limitations"))
    for limitation in DEFAULT_LIMITATIONS:
        if limitation not in limitations:
            limitations.append(limitation)
    normalized["limitations"] = limitations

    normalized["warning"] = WARNING_TEXT
    normalized.setdefault("model_name", "unknown")
    normalized.setdefault("prompt_version", "unknown")
    try:
        latency_ms = int(normalized.get("latency_ms", 0))
    except (TypeError, ValueError):
        latency_ms = 0
    normalized["latency_ms"] = max(latency_ms, 0)

    return normalized


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except (TypeError, ValueError):
        errors.append("confidence is not numeric")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    raw_valid, raw_errors = validate_prediction(pred)

    normalized = normalize_prediction(pred)
    normalized_valid, normalized_errors = validate_prediction(normalized)

    errors = raw_errors + [error for error in normalized_errors if error not in raw_errors]

    if not raw_valid or not normalized_valid:
        if "guardrail triggered: invalid output schema" not in normalized["limitations"]:
            normalized["limitations"].append("guardrail triggered: invalid output schema")

    if normalized["image_quality"] == "poor":
        normalized["predicted_class"] = "uncertain"
        normalized["confidence"] = min(normalized["confidence"], 0.50)
        if "Qualité image insuffisante pour une sortie fiable." not in normalized["limitations"]:
            normalized["limitations"].append("Qualité image insuffisante pour une sortie fiable.")

    if normalized["confidence"] < 0.60:
        normalized["predicted_class"] = "uncertain"
        if "Confiance insuffisante pour proposer une classe autre que uncertain." not in normalized["limitations"]:
            normalized["limitations"].append("Confiance insuffisante pour proposer une classe autre que uncertain.")

    normalized["warning"] = WARNING_TEXT
    normalized["guardrail_errors"] = errors

    return normalized
