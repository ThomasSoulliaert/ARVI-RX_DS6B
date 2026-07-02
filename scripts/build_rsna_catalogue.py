from __future__ import annotations

"""Build a case catalogue from the RSNA Pneumonia Detection Challenge dataset,
in the same CSV format as data/synthetic_cases.csv, so it can be consumed by
eval/run_evaluation.py exactly like the synthetic dataset.

Label mapping (documented decision, see docs/evaluation_protocol.md):
    "Normal"                       -> normal
    "Lung Opacity"                 -> suspected_opacity
    "No Lung Opacity / Not Normal" -> EXCLUDED from the scored catalogue.
        These images show some abnormality that is not a lung opacity (e.g.
        cardiomegaly, effusion, other findings). They are neither "normal" nor
        "suspected_opacity" for this task, and "uncertain" is a model safety
        output, not a dataset ground truth label, so we do not force a label
        onto them. They are written to a separate, unscored file
        (rsna_excluded_not_normal.csv) for later qualitative review only.

"uncertain" is never used as a ground-truth label here for the same reason.

Sampling is stratified (balanced normal / suspected_opacity) and reproducible
via a fixed seed. The "smoke" and "dev" splits are disjoint.

Requires data/rsna/ to already be populated by scripts/download_rsna.py.

Usage:
    python scripts/build_rsna_catalogue.py [--rsna-dir data/rsna] [--smoke-n 20]
        [--dev-n 150] [--seed 0]
"""

import argparse
import csv
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing import compute_image_quality, dicom_to_png, open_image  # noqa: E402

LABEL_MAP = {
    "Normal": "normal",
    "Lung Opacity": "suspected_opacity",
    "No Lung Opacity / Not Normal": None,  # excluded, see module docstring
}
SCORED_LABELS = ["normal", "suspected_opacity"]
CSV_FIELDNAMES = ["case_id", "image_path", "source", "label", "split", "quality", "notes"]


def map_class_to_label(rsna_class: str) -> str | None:
    """Map an RSNA `class` string to our 3-class label, or None if excluded."""
    if rsna_class not in LABEL_MAP:
        raise ValueError(f"Unknown RSNA class: {rsna_class!r}")
    return LABEL_MAP[rsna_class]


def read_targets(labels_csv: Path) -> dict[str, int]:
    """One row per patientId: RSNA has one label row per bounding box, so a
    patient with an opacity can appear multiple times. Target is the same
    value (1) across a patient's rows in that case, so max() is a safe reduce."""
    targets: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["patientId"]
            targets[pid] = max(targets.get(pid, 0), int(row["Target"]))
    return targets


def read_classes(class_csv: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    with class_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classes.setdefault(row["patientId"], row["class"])
    return classes


def balanced_sample(pool_by_label: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    """Sample n patientIds as evenly as possible across labels in pool_by_label.
    Mutates the pools in place (removes sampled ids) so a later call draws a
    disjoint set."""
    labels = list(pool_by_label)
    per_label = n // len(labels)
    remainder = n % len(labels)
    chosen: list[str] = []
    for i, label in enumerate(labels):
        take = per_label + (1 if i < remainder else 0)
        pool = pool_by_label[label]
        take = min(take, len(pool))
        picked = rng.sample(pool, take)
        chosen.extend(picked)
        pool_by_label[label] = [pid for pid in pool if pid not in set(picked)]
    return chosen


def build_catalogue(
    rsna_dir: Path,
    smoke_n: int,
    dev_n: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    targets = read_targets(rsna_dir / "stage_2_train_labels.csv")
    classes = read_classes(rsna_dir / "stage_2_detailed_class_info.csv")
    images_dir = rsna_dir / "stage_2_train_images"
    png_dir = rsna_dir / "images_png"

    scored_pool: dict[str, list[str]] = {label: [] for label in SCORED_LABELS}
    excluded_rows: list[dict] = []

    for pid in sorted(set(targets) & set(classes)):
        rsna_class = classes[pid]
        label = map_class_to_label(rsna_class)
        if label is None:
            excluded_rows.append(
                {
                    "case_id": f"RSNA_{pid}",
                    "image_path": "",
                    "source": "rsna_real",
                    "label": "",
                    "split": "excluded",
                    "quality": "",
                    "notes": f"excluded: rsna class={rsna_class!r}, Target={targets[pid]}",
                }
            )
            continue
        scored_pool[label].append(pid)

    rng = random.Random(seed)
    smoke_ids = balanced_sample(scored_pool, smoke_n, rng)
    dev_ids = balanced_sample(scored_pool, dev_n, rng)

    rows: list[dict] = []
    for split, pids in (("smoke", smoke_ids), ("dev", dev_ids)):
        for pid in pids:
            label = "normal" if classes[pid] == "Normal" else "suspected_opacity"
            dicom_path = images_dir / f"{pid}.dcm"
            png_path = png_dir / f"{pid}.png"
            dicom_to_png(dicom_path, png_path)
            quality = compute_image_quality(open_image(png_path))["quality"]
            try:
                image_path = png_path.relative_to(ROOT)
            except ValueError:
                image_path = png_path  # rsna_dir is outside the repo root (e.g. in tests)
            rows.append(
                {
                    "case_id": f"RSNA_{pid}",
                    "image_path": str(image_path).replace("\\", "/"),
                    "source": "rsna_real",
                    "label": label,
                    "split": split,
                    "quality": quality,
                    "notes": f"RSNA class={classes[pid]!r}, Target={targets[pid]}",
                }
            )
    return rows, excluded_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsna-dir", type=Path, default=ROOT / "data" / "rsna")
    parser.add_argument("--smoke-n", type=int, default=20)
    parser.add_argument("--dev-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, excluded_rows = build_catalogue(args.rsna_dir, args.smoke_n, args.dev_n, args.seed)

    catalogue_path = args.rsna_dir / "rsna_catalogue.csv"
    excluded_path = args.rsna_dir / "rsna_excluded_not_normal.csv"
    write_csv(catalogue_path, rows)
    write_csv(excluded_path, excluded_rows)

    print(f"Wrote {len(rows)} scored cases to {catalogue_path}")
    print(f"Wrote {len(excluded_rows)} excluded 'No Lung Opacity / Not Normal' cases to {excluded_path}")


if __name__ == "__main__":
    main()
