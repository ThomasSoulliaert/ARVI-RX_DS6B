from __future__ import annotations

"""Generate a small 'hard' synthetic eval set that stresses the real pixel-based
image-quality checks in src/preprocessing.compute_image_quality (brightness,
contrast, size), independent of the filename shortcuts used elsewhere in the
toy pipeline.

Unlike data/sample_images/*, these filenames never contain "uncertain" or
"limited", so src.preprocessing.basic_quality_flag cannot cheat via the
filename override: the quality flag it returns is driven only by real pixel
statistics. This lets baseline vs improved be compared on genuine
good/medium/poor quality cases instead of a dataset that is already at a
100% ceiling.

Run: python eval/generate_hard_cases.py
"""

from pathlib import Path
import csv

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "hard_images"
CSV_PATH = ROOT / "data" / "hard_cases.csv"

RNG = np.random.default_rng(0)


def _save(name: str, arr: np.ndarray) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
    return path


def _base(mean: float, std: float, size: int = 512) -> np.ndarray:
    arr = RNG.normal(mean, std, (size, size))
    return np.clip(arr, 0, 255)


def _with_opacity_patch(arr: np.ndarray, boost: float = 55.0) -> np.ndarray:
    arr = arr.copy()
    size = arr.shape[0]
    r0, r1 = int(size * 0.35), int(size * 0.55)
    c0, c1 = int(size * 0.55), int(size * 0.75)
    arr[r0:r1, c0:c1] = np.clip(arr[r0:r1, c0:c1] + boost, 0, 255)
    return arr


CASES = []


def add_case(case_id: str, filename: str, label: str, quality_bucket: str, arr: np.ndarray, notes: str) -> None:
    path = _save(filename, arr)
    CASES.append(
        {
            "case_id": case_id,
            "image_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source": "synthetic_hard",
            "label": label,
            "split": "hard",
            "quality": quality_bucket,
            "notes": notes,
        }
    )


# --- good quality (0 issues): clear signal, both modes should agree with baseline label ---
add_case(
    "CXR_HARD_001", "CXR_HARD_001_normal_good.png", "normal", "good",
    _base(mean=130, std=22), "good quality, no opacity marker",
)
add_case(
    "CXR_HARD_002", "CXR_HARD_002_suspected_opacity_good.png", "suspected_opacity", "good",
    _with_opacity_patch(_base(mean=130, std=22)), "good quality, clear opacity patch",
)
add_case(
    "CXR_HARD_003", "CXR_HARD_003_normal_good.png", "normal", "good",
    _base(mean=140, std=24), "good quality, no opacity marker",
)
add_case(
    "CXR_HARD_004", "CXR_HARD_004_suspected_opacity_good.png", "suspected_opacity", "good",
    _with_opacity_patch(_base(mean=140, std=24)), "good quality, clear opacity patch",
)

# --- medium quality (exactly 1 real issue: low contrast) : class signal still readable,
#     ground truth stays the true class -> tests whether a mode over-triggers "uncertain". ---
add_case(
    "CXR_HARD_005", "CXR_HARD_005_normal_lowcontrast.png", "normal", "medium",
    _base(mean=120, std=8), "single issue: low contrast only, true class normal",
)
add_case(
    "CXR_HARD_006", "CXR_HARD_006_suspected_opacity_lowcontrast.png", "suspected_opacity", "medium",
    _with_opacity_patch(_base(mean=120, std=6), boost=12), "single issue: low contrast only, true class suspected_opacity",
)
add_case(
    "CXR_HARD_007", "CXR_HARD_007_normal_lowcontrast.png", "normal", "medium",
    _base(mean=110, std=9), "single issue: low contrast only, true class normal",
)
add_case(
    "CXR_HARD_008", "CXR_HARD_008_suspected_opacity_lowcontrast.png", "suspected_opacity", "medium",
    _with_opacity_patch(_base(mean=110, std=6), boost=12), "single issue: low contrast only, true class suspected_opacity",
)

# --- poor quality (2+ real issues: dark + low contrast): image itself is unreliable,
#     ground truth is "uncertain" regardless of the filename class hint. ---
add_case(
    "CXR_HARD_009", "CXR_HARD_009_normal_dim.png", "uncertain", "poor",
    _base(mean=10, std=6), "two issues: too dark + low contrast -> unreliable image",
)
add_case(
    "CXR_HARD_010", "CXR_HARD_010_suspected_opacity_dim.png", "uncertain", "poor",
    _with_opacity_patch(_base(mean=10, std=6), boost=15), "two issues: too dark + low contrast -> unreliable image",
)
add_case(
    "CXR_HARD_011", "CXR_HARD_011_normal_dim.png", "uncertain", "poor",
    _base(mean=8, std=5), "two issues: too dark + low contrast -> unreliable image",
)
add_case(
    "CXR_HARD_012", "CXR_HARD_012_suspected_opacity_dim.png", "uncertain", "poor",
    _with_opacity_patch(_base(mean=8, std=5), boost=15), "two issues: too dark + low contrast -> unreliable image",
)


def main() -> None:
    fieldnames = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(CASES)
    print(f"Wrote {len(CASES)} cases to {CSV_PATH}")


if __name__ == "__main__":
    main()
