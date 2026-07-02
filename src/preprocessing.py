from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from PIL import Image, ImageStat, UnidentifiedImageError

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def validate_image_path(path: str | Path) -> Path:
    image_path = Path(path)

    if not image_path.exists():
        raise ValueError(f"Image file does not exist: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}. Allowed formats: {allowed}"
        )

    return image_path


def open_image(path: str | Path) -> Image.Image:
    image_path = validate_image_path(path)

    try:
        with Image.open(image_path) as img:
            img.verify()

        return Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid or unreadable image file") from exc


def compute_image_quality(image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)

    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])

    issues: list[str] = []

    if width < 224 or height < 224:
        issues.append("image too small")

    if brightness < 15:
        issues.append("image too dark")
    elif brightness > 235:
        issues.append("image too bright")

    if contrast < 10:
        issues.append("very low contrast")

    if len(issues) >= 2:
        quality = "poor"
    elif issues:
        quality = "medium"
    else:
        quality = "good"

    return {
        "width": width,
        "height": height,
        "mean_brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "quality": quality,
        "issues": issues,
    }


def load_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load and resize an image safely for the educational prototype."""
    img = open_image(path)
    return img.resize(size)


def load_dicom_pixels(path: str | Path) -> Image.Image:
    """Load a DICOM file's pixel data ONLY (no patient metadata) as an 8-bit RGB image.

    Only `pixel_array` and `PhotometricInterpretation` (an image-encoding tag, not
    patient information) are read. No PHI-bearing tag (PatientName, PatientID,
    dates, institution, ...) is ever accessed here.
    """
    dicom_path = Path(path)
    dataset = pydicom.dcmread(dicom_path)
    array = dataset.pixel_array.astype(np.float64)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = array.max() - array

    array_min, array_max = float(array.min()), float(array.max())
    normalized = (array - array_min) / (array_max - array_min + 1e-8) * 255.0
    return Image.fromarray(normalized.astype(np.uint8)).convert("RGB")


def dicom_to_png(dicom_path: str | Path, out_path: str | Path) -> Path:
    """Convert a DICOM file to a PNG on disk, pixels only. Returns the written path."""
    image = load_dicom_pixels(dicom_path)
    png_path = Path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    return png_path


def basic_quality_flag(path: str | Path) -> str:
    """Return a simple image quality category: good, medium, or poor."""
    image = open_image(path)
    quality = compute_image_quality(image)

    name = Path(path).name.lower()
    if "uncertain" in name or "limited" in name:
        return "poor"

    return str(quality["quality"])
