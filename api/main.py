from __future__ import annotations

import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.pipeline import run_prediction

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.1.0")
UPLOAD_DIR = Path("tmp_uploads")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("improved"),
) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"

    try:
        with target.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return run_prediction(str(target), mode=mode, save=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed in the educational prototype.",
        ) from exc
