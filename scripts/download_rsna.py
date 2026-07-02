from __future__ import annotations

"""Download the RSNA Pneumonia Detection Challenge dataset from Kaggle.

Requires the `kaggle` package and valid credentials (either ~/.kaggle/kaggle.json
or the KAGGLE_USERNAME / KAGGLE_KEY environment variables) to already be
configured on the machine running this script -- see
https://www.kaggle.com/docs/api for how to create a token.

This script only downloads and unzips the competition archive; it does not
build the case catalogue (see build_rsna_catalogue.py for that).

Usage:
    python scripts/download_rsna.py [--out-dir data/rsna] [--force]
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-pneumonia-detection-challenge"
EXPECTED_ENTRIES = [
    "stage_2_train_labels.csv",
    "stage_2_detailed_class_info.csv",
    "stage_2_train_images",
]


def already_downloaded(out_dir: Path) -> bool:
    return all((out_dir / entry).exists() for entry in EXPECTED_ENTRIES)


def download(out_dir: Path, force: bool) -> None:
    if not force and already_downloaded(out_dir):
        print(f"Already present in {out_dir}, skipping download (use --force to re-download).")
        return

    if shutil.which("kaggle") is None:
        print(
            "The 'kaggle' CLI is not on PATH. Install it with `pip install kaggle` and "
            "configure credentials (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) before "
            "re-running this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{COMPETITION}.zip"

    print(f"Downloading {COMPETITION} into {out_dir} ...")
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(out_dir),
        ],
        check=True,
    )

    if not archive_path.exists():
        # Some kaggle CLI versions name the archive differently; fall back to the
        # first .zip found in out_dir.
        zips = list(out_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No archive found in {out_dir} after download.")
        archive_path = zips[0]

    print(f"Unzipping {archive_path.name} ...")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "rsna")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    args = parser.parse_args()
    download(args.out_dir, args.force)
    print("Done. RSNA data is under", args.out_dir, "(gitignored, never commit it).")


if __name__ == "__main__":
    main()
