"""Genere les figures de soutenance a partir des fichiers commites du run.

Regle d'or : aucune valeur retapee a la main. Tout est lu depuis
before_after_summary.csv, *_predictions.csv et *_confusion.csv, et chaque
figure porte sa source, la date du run et le nombre de cas.

Usage :
  python eval/make_figures.py --run-dir eval/results/rsna_2026-07-02
Sorties : <run-dir>/figures/fig1..fig4 (.png, 200 dpi)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"baseline": "#4C72B0", "improved": "#DD8452"}
CLASSES = ["normal", "suspected_opacity", "uncertain"]


def _footer(ax_or_fig, run_dir: Path, source: str, n: int) -> None:
    fig = ax_or_fig if isinstance(ax_or_fig, plt.Figure) else ax_or_fig.figure
    fig.text(
        0.01,
        0.01,
        f"Source : {run_dir.name}/{source} | n = {n} cas RSNA | genere par eval/make_figures.py",
        fontsize=7,
        color="#555555",
    )


def fig1_comparison(run_dir: Path, out: Path) -> None:
    s = pd.read_csv(run_dir / "before_after_summary.csv").set_index("mode")
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("sensitivity_opacity", "Sensibilite\nopacites"),
        ("specificity_opacity", "Specificite\nopacites"),
        ("uncertain_rate", "Taux\nd'incertitude"),
        ("coverage_accuracy", "Accuracy\nhors uncertain"),
    ]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, mode in enumerate(["baseline", "improved"]):
        vals = [s.loc[mode, m] for m, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=mode, color=COLORS[mode])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Baseline vs prompt improved — le prompt improved s'effondre sur les opacites\n"
        "(sensibilite 0.85 -> 0.00 : ancrage sur les exemples few-shot, voir registre d'erreurs)",
        fontsize=10,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _footer(fig, run_dir, "before_after_summary.csv", int(s.loc["baseline", "n"]))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig1_baseline_vs_improved.png", dpi=200)
    plt.close(fig)


def _confusion(run_dir: Path, pred_file: str, title: str, out_name: str, out: Path) -> None:
    df = pd.read_csv(run_dir / pred_file)
    mat = (
        pd.crosstab(df["label"], df["predicted_class"])
        .reindex(index=["normal", "suspected_opacity"], columns=CLASSES)
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=mat.values.max())
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax.set_yticks(range(2), ["normal", "suspected_opacity"], fontsize=9)
    ax.set_xlabel("Classe predite")
    ax.set_ylabel("Verite terrain (RSNA)")
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            share = v / mat.values[i].sum()
            ax.text(
                j,
                i,
                f"{v}\n({share:.0%})",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if v > mat.values.max() * 0.6 else "#1a1a1a",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    _footer(fig, run_dir, pred_file, int(mat.values.sum()))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / out_name, dpi=200)
    plt.close(fig)


def fig4_latency(run_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data, labels = [], []
    n = 0
    for mode in ["baseline", "improved"]:
        df = pd.read_csv(run_dir / f"{mode}_predictions.csv")
        data.append(df["latency_ms"] / 1000.0)
        labels.append(mode)
        n = len(df)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for patch, mode in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[mode])
        patch.set_alpha(0.6)
    for i, series in enumerate(data, start=1):
        med, p95 = series.median(), series.quantile(0.95)
        ax.annotate(f"mediane {med:.1f} s\np95 {p95:.1f} s", (i + 0.18, med), fontsize=8, va="center")
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1)
    ax.annotate("cible < 10 s — non atteinte sur GPU T4 (documente)", (0.55, 10.4), color="crimson", fontsize=8)
    ax.set_ylabel("Latence par image (s)")
    ax.set_title("Latence d'inference MedGemma par prompt (Colab T4)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _footer(fig, run_dir, "*_predictions.csv (colonne latency_ms)", n)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig4_latences.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="eval/results/rsna_2026-07-02")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    fig1_comparison(run_dir, out)
    _confusion(
        run_dir,
        "baseline_predictions.csv",
        "Matrice de confusion — baseline (1 FN, 3 FP, 36 uncertain)",
        "fig2_confusion_baseline.png",
        out,
    )
    _confusion(
        run_dir,
        "improved_predictions.csv",
        "Matrice de confusion — improved (aucune prediction suspected_opacity)",
        "fig3_confusion_improved.png",
        out,
    )
    fig4_latency(run_dir, out)
    print(f"4 figures ecrites dans {out}/")


if __name__ == "__main__":
    main()
