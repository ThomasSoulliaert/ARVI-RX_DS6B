from __future__ import annotations

"""Construit la matrice de confusion 3 classes depuis un CSV de prédictions.

Le CSV d'entrée est celui produit par eval/run_evaluation.py
(colonnes minimales : label, predicted_class). La matrice est calculée depuis
le fichier de résultats — jamais saisie à la main — comme l'exige le protocole.

Usage:
    python eval/build_confusion_matrix.py eval/results/rsna_2026-07-02/improved_predictions.csv
    # options : --out-png chemin.png --out-csv chemin.csv --title "Improved (dev RSNA)"

Sorties : matrice affichée en texte, CSV des effectifs, PNG (si matplotlib
est installé). L'ordre des classes est constant : normal, suspected_opacity,
uncertain (lignes = vérité terrain, colonnes = prédiction).
"""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.metrics import CLASSES, accuracy, macro_f1, one_vs_rest_rates  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_matrix(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    """matrix[i][j] = nombre de cas de vérité CLASSES[i] prédits CLASSES[j]."""
    index = {c: i for i, c in enumerate(CLASSES)}
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def print_matrix(matrix: list[list[int]]) -> None:
    col_width = max(len(c) for c in CLASSES) + 2
    header = "vérité \\ prédit".ljust(col_width) + "".join(c.rjust(col_width) for c in CLASSES)
    print(header)
    for label, row in zip(CLASSES, matrix):
        print(label.ljust(col_width) + "".join(str(v).rjust(col_width) for v in row))


def write_matrix_csv(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth \\ predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix):
            writer.writerow([label, *row])


def save_matrix_png(path: Path, matrix: list[list[int]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")  # pas d'affichage interactif requis (CI, Colab)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(title)
    vmax = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > vmax / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    rows = read_rows(args.predictions_csv)
    if not rows:
        raise SystemExit(f"Aucune ligne dans {args.predictions_csv}")
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    matrix = build_matrix(y_true, y_pred)

    title = args.title or args.predictions_csv.stem
    print(f"Matrice de confusion — {title} ({len(rows)} cas)\n")
    print_matrix(matrix)
    rates = one_vs_rest_rates(y_true, y_pred)
    print(f"\naccuracy : {accuracy(y_true, y_pred):.4f}")
    print(f"macro-F1 : {macro_f1(y_true, y_pred):.4f}")
    for cls in CLASSES:
        print(
            f"{cls}: sensibilité={rates[cls]['sensitivity']:.4f} "
            f"spécificité={rates[cls]['specificity']:.4f} (one-vs-rest)"
        )

    out_csv = args.out_csv or args.predictions_csv.with_name(
        args.predictions_csv.stem + "_confusion.csv"
    )
    write_matrix_csv(out_csv, matrix)
    print(f"\nMatrice CSV : {out_csv}")

    out_png = args.out_png or args.predictions_csv.with_name(
        args.predictions_csv.stem + "_confusion.png"
    )
    if save_matrix_png(out_png, matrix, f"Matrice de confusion — {title}"):
        print(f"Matrice PNG : {out_png}")
    else:
        print("matplotlib non installé : PNG non généré (pip install matplotlib).")


if __name__ == "__main__":
    main()
