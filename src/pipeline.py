from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from .database import DEFAULT_DB_PATH, insert_run
from .guardrails import apply_safety_guardrails
from .inference import PROMPT_MODES, predict_with_model
from .preprocessing import compute_image_quality, open_image, validate_image_path

ALLOWED_MODES = PROMPT_MODES
INFERENCE_SIZE = (512, 512)


def _build_case_id(image_path: Path, case_id: str | None) -> str:
    if case_id:
        return case_id
    return f"{image_path.stem}_{uuid.uuid4().hex[:8]}"


def run_prediction(
    image_path: str,
    mode: str = "improved",
    case_id: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run the educational prediction flow and return a safe JSON dictionary."""
    start = time.perf_counter()

    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode. Expected one of {sorted(ALLOWED_MODES)}.")

    image_file = validate_image_path(image_path)
    resolved_case_id = _build_case_id(image_file, case_id)

    image = open_image(image_file)
    preprocessed_image = image.resize(INFERENCE_SIZE)
    quality_metrics = compute_image_quality(image)
    quality_metrics["inference_width"] = preprocessed_image.width
    quality_metrics["inference_height"] = preprocessed_image.height

    raw_prediction = predict_with_model(image_file, mode=mode)
    raw_prediction["image_quality"] = quality_metrics["quality"]
    raw_prediction["mode"] = mode
    raw_prediction["case_id"] = resolved_case_id
    raw_prediction["image_path"] = str(image_file)
    raw_prediction["preprocessing"] = quality_metrics

    prediction = apply_safety_guardrails(raw_prediction)
    prediction["latency_ms"] = int((time.perf_counter() - start) * 1000)
    prediction["case_id"] = resolved_case_id
    prediction["image_path"] = str(image_file)
    prediction["mode"] = mode
    prediction["preprocessing"] = quality_metrics

    if save:
        try:
            insert_run(DEFAULT_DB_PATH, resolved_case_id, str(image_file), prediction)
            prediction["saved"] = True
        except Exception as exc:
            prediction["saved"] = False
            prediction.setdefault("limitations", []).append(
                f"SQLite logging failed: {exc}"
            )

    return prediction
