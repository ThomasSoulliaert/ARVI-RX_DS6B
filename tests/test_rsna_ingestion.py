from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from scripts.build_rsna_catalogue import balanced_sample, build_catalogue, map_class_to_label
from src.preprocessing import dicom_to_png, load_dicom_pixels


def _make_dicom(path: Path, rows: int = 32, cols: int = 32, photometric: str = "MONOCHROME2") -> np.ndarray:
    """Write a minimal but valid DICOM file (pixels only matter for our code path).

    Includes PatientName/PatientID to prove load_dicom_pixels() does not need
    them and never surfaces them.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "Test^PHI"
    ds.PatientID = "PHI123"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    array = np.tile(np.linspace(0, 255, cols, dtype=np.uint8), (rows, 1))
    ds.PixelData = array.astype(np.uint8).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)
    return array


def test_map_class_to_label_known_classes() -> None:
    assert map_class_to_label("Normal") == "normal"
    assert map_class_to_label("Lung Opacity") == "suspected_opacity"
    assert map_class_to_label("No Lung Opacity / Not Normal") is None


def test_map_class_to_label_rejects_unknown_class() -> None:
    with pytest.raises(ValueError):
        map_class_to_label("Something Else")


def test_balanced_sample_is_deterministic_and_balanced() -> None:
    pool = {"normal": [f"n{i}" for i in range(10)], "suspected_opacity": [f"o{i}" for i in range(10)]}
    rng1 = __import__("random").Random(0)
    rng2 = __import__("random").Random(0)
    picked1 = balanced_sample({k: list(v) for k, v in pool.items()}, 6, rng1)
    picked2 = balanced_sample({k: list(v) for k, v in pool.items()}, 6, rng2)

    assert picked1 == picked2
    assert len(picked1) == 6
    assert sum(pid.startswith("n") for pid in picked1) == 3
    assert sum(pid.startswith("o") for pid in picked1) == 3


def test_balanced_sample_disjoint_across_successive_calls() -> None:
    pool = {"normal": [f"n{i}" for i in range(10)], "suspected_opacity": [f"o{i}" for i in range(10)]}
    rng = __import__("random").Random(0)
    first = balanced_sample(pool, 6, rng)
    second = balanced_sample(pool, 6, rng)

    assert set(first).isdisjoint(second)


def test_load_dicom_pixels_returns_rgb_image_from_pixels_only(tmp_path: Path) -> None:
    dicom_path = tmp_path / "case.dcm"
    array = _make_dicom(dicom_path)

    image = load_dicom_pixels(dicom_path)

    assert image.mode == "RGB"
    assert image.size == (array.shape[1], array.shape[0])
    # Pixel values should still be monotonically increasing across columns
    # like the source array (min-max normalization preserves ordering).
    row = np.array(image.convert("L"))[0]
    assert (np.diff(row.astype(int)) >= 0).all()


def test_load_dicom_pixels_inverts_monochrome1(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.dcm"
    inverted_path = tmp_path / "inverted.dcm"
    _make_dicom(normal_path, photometric="MONOCHROME2")
    _make_dicom(inverted_path, photometric="MONOCHROME1")

    normal_row = np.array(load_dicom_pixels(normal_path).convert("L"))[0]
    inverted_row = np.array(load_dicom_pixels(inverted_path).convert("L"))[0]

    # MONOCHROME1 is stored the same but should be photometrically inverted
    # by load_dicom_pixels, so the gradient direction flips.
    assert (np.diff(normal_row.astype(int)) >= 0).all()
    assert (np.diff(inverted_row.astype(int)) <= 0).all()


def test_dicom_to_png_writes_file(tmp_path: Path) -> None:
    dicom_path = tmp_path / "case.dcm"
    _make_dicom(dicom_path)
    png_path = tmp_path / "out" / "case.png"

    result = dicom_to_png(dicom_path, png_path)

    assert result == png_path
    assert png_path.exists()


def _write_rsna_fixture(rsna_dir: Path) -> None:
    images_dir = rsna_dir / "stage_2_train_images"
    images_dir.mkdir(parents=True)
    patients = {
        "p_normal_1": "Normal",
        "p_normal_2": "Normal",
        "p_opacity_1": "Lung Opacity",
        "p_opacity_2": "Lung Opacity",
        "p_other_1": "No Lung Opacity / Not Normal",
    }
    for pid in patients:
        _make_dicom(images_dir / f"{pid}.dcm")

    with (rsna_dir / "stage_2_train_labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patientId", "x", "y", "width", "height", "Target"])
        for pid, cls in patients.items():
            target = 1 if cls == "Lung Opacity" else 0
            writer.writerow([pid, "", "", "", "", target])

    with (rsna_dir / "stage_2_detailed_class_info.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patientId", "class"])
        for pid, cls in patients.items():
            writer.writerow([pid, cls])


def test_build_catalogue_excludes_not_normal_and_balances_splits(tmp_path: Path) -> None:
    rsna_dir = tmp_path / "rsna"
    _write_rsna_fixture(rsna_dir)

    rows, excluded_rows = build_catalogue(rsna_dir, smoke_n=2, dev_n=2, seed=0)

    assert len(excluded_rows) == 1
    assert excluded_rows[0]["case_id"] == "RSNA_p_other_1"

    assert len(rows) == 4
    assert {r["label"] for r in rows} == {"normal", "suspected_opacity"}
    assert {r["split"] for r in rows} == {"smoke", "dev"}
    for row in rows:
        assert Path(row["image_path"]).name.endswith(".png")
        assert row["quality"] in {"good", "medium", "poor"}

    smoke_labels = [r["label"] for r in rows if r["split"] == "smoke"]
    dev_labels = [r["label"] for r in rows if r["split"] == "dev"]
    assert sorted(smoke_labels) == ["normal", "suspected_opacity"]
    assert sorted(dev_labels) == ["normal", "suspected_opacity"]
