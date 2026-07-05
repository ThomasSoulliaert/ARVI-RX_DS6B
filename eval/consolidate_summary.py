"""Consolide les métriques par prompt en un seul before_after_summary.csv.

Contexte : `run_evaluation.py` écrit un `before_after_summary.csv` limité aux
prompts évalués dans un même appel (baseline + improved), et un run séparé
(`--prompts improved_v2`) a produit ses métriques à part. Ce script relit la
source de vérité — les fichiers `*_metrics.json` commités du run — et régénère
un `before_after_summary.csv` unique couvrant les trois prompts, dans un ordre
canonique. Aucune valeur n'est retapée : tout vient des JSON du run.

Usage :
  python eval/consolidate_summary.py --run-dir eval/results/rsna_2026-07-02
Sortie : <run-dir>/before_after_summary.csv (baseline, improved, improved_v2).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Ordre d'affichage voulu (baseline d'abord, puis les améliorations).
# Tout mode absent de cette liste est ajouté à la fin, trié alphabétiquement.
PREFERRED_ORDER = ["baseline", "improved", "improved_v2"]

# Colonnes du résumé, dans l'ordre produit par run_evaluation.py.
COLUMNS = [
    "mode", "n", "accuracy", "macro_f1", "sensitivity_opacity",
    "specificity_opacity", "macro_sensitivity", "macro_specificity",
    "json_valid_rate", "warning_rate", "uncertain_rate", "coverage_accuracy",
    "coverage_n", "latency_median_ms", "latency_p95_ms",
]


def _mode_from_filename(path: Path) -> str:
    # baseline_metrics.json -> baseline, improved_v2_metrics.json -> improved_v2
    return path.name[: -len("_metrics.json")]


def collect_rows(run_dir: Path) -> list[dict]:
    metrics_files = sorted(run_dir.glob("*_metrics.json"))
    if not metrics_files:
        raise SystemExit(f"Aucun *_metrics.json dans {run_dir}")

    by_mode = {_mode_from_filename(p): json.loads(p.read_text(encoding="utf-8")) for p in metrics_files}
    ordered = [m for m in PREFERRED_ORDER if m in by_mode]
    ordered += sorted(m for m in by_mode if m not in PREFERRED_ORDER)

    rows = []
    for mode in ordered:
        row = {"mode": mode, **by_mode[mode]}
        rows.append({col: row.get(col, "") for col in COLUMNS})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="eval/results/rsna_2026-07-02")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = collect_rows(run_dir)
    out = run_dir / "before_after_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} modes consolidés dans {out} : {', '.join(r['mode'] for r in rows)}")


if __name__ == "__main__":
    main()
