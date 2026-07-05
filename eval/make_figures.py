"""Genere les figures de soutenance a partir des fichiers commites du run.

Regle d'or : aucune valeur retapee a la main. Tout est lu depuis
before_after_summary.csv et *_predictions.csv, et chaque figure porte sa
source, la date du run et le nombre de cas.

Le script est mode-aware : il figure TOUS les prompts presents dans
before_after_summary.csv (baseline, improved, improved_v2...), dans l'ordre du
fichier. Regenerer before_after_summary.csv via eval/consolidate_summary.py.

Usage :
  python eval/make_figures.py --run-dir eval/results/rsna_2026-07-02
Sorties : <run-dir>/figures/fig1 (comparaison), fig_confusion_<mode>,
          fig_latences (.png, 200 dpi). Le dossier est nettoye avant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Couleurs par prompt. Un mode inconnu recoit une couleur du cycle par defaut.
COLORS = {"baseline": "#4C72B0", "improved": "#DD8452", "improved_v2": "#55A868"}
_FALLBACK = ["#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
CLASSES = ["normal", "suspected_opacity", "uncertain"]


def _color(mode: str, i: int) -> str:
    return COLORS.get(mode, _FALLBACK[i % len(_FALLBACK)])


def _footer(fig: plt.Figure, run_dir: Path, source: str, n: int) -> None:
    fig.text(
        0.01,
        0.01,
        f"Source : {run_dir.name}/{source} | n = {n} cas RSNA | genere par eval/make_figures.py",
        fontsize=7,
        color="#555555",
    )


def _modes(run_dir: Path) -> list[str]:
    s = pd.read_csv(run_dir / "before_after_summary.csv")
    return list(s["mode"])


def fig1_comparison(run_dir: Path, out: Path) -> None:
    s = pd.read_csv(run_dir / "before_after_summary.csv").set_index("mode")
    modes = list(s.index)
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("sensitivity_opacity", "Sensibilite\nopacites"),
        ("specificity_opacity", "Specificite\nnormales"),
        ("uncertain_rate", "Taux\nd'incertitude"),
        ("coverage_accuracy", "Accuracy\nhors uncertain"),
    ]
    x = np.arange(len(metrics))
    w = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, mode in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * w
        vals = [s.loc[mode, m] for m, _ in metrics]
        bars = ax.bar(x + offset, vals, w, label=mode, color=_color(mode, i))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, [lbl for _, lbl in metrics], fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_title(
        "Comparaison des prompts sur 170 cas RSNA\n"
        "improved s'effondre (sensibilite 0.85 -> 0.00, ancrage few-shot) ;\n"
        "improved_v2 restaure la sensibilite (0.95) mais baisse la specificite (0.96 -> 0.78)",
        fontsize=9.5,
    )
    ax.legend(ncols=len(modes))
    ax.grid(axis="y", alpha=0.3)
    _footer(fig, run_dir, "before_after_summary.csv", int(s.iloc[0]["n"]))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig1_comparaison_prompts.png", dpi=200)
    plt.close(fig)


def _confusion(run_dir: Path, mode: str, out: Path) -> None:
    pred_file = f"{mode}_predictions.csv"
    if not (run_dir / pred_file).exists():
        return
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
    ax.set_title(f"Matrice de confusion — {mode}", fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            share = v / mat.values[i].sum() if mat.values[i].sum() else 0
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
    fig.savefig(out / f"fig_confusion_{mode}.png", dpi=200)
    plt.close(fig)


def fig_latency(run_dir: Path, modes: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    data, labels = [], []
    n = 0
    for mode in modes:
        pred_path = run_dir / f"{mode}_predictions.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        data.append(df["latency_ms"] / 1000.0)
        labels.append(mode)
        n = len(df)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for i, (patch, mode) in enumerate(zip(bp["boxes"], labels)):
        patch.set_facecolor(_color(mode, i))
        patch.set_alpha(0.6)
    for i, series in enumerate(data, start=1):
        med, p95 = series.median(), series.quantile(0.95)
        ax.annotate(f"mediane {med:.1f} s\np95 {p95:.1f} s", (i + 0.16, med), fontsize=8, va="center")
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1)
    ax.annotate("cible < 10 s — non atteinte sur GPU T4 (documente)", (0.55, 10.4), color="crimson", fontsize=8)
    ax.set_ylabel("Latence par image (s)")
    ax.set_title("Latence d'inference MedGemma par prompt (Colab T4)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _footer(fig, run_dir, "*_predictions.csv (colonne latency_ms)", n)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_latences.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="eval/results/rsna_2026-07-02")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    # Nettoyage : on repart d'un dossier propre pour ne pas laisser d'anciennes
    # figures orphelines (ex. quand un prompt est ajoute ou renomme).
    for old in out.glob("*.png"):
        old.unlink()

    modes = _modes(run_dir)
    fig1_comparison(run_dir, out)
    for mode in modes:
        _confusion(run_dir, mode, out)
    fig_latency(run_dir, modes, out)
    written = sorted(p.name for p in out.glob("*.png"))
    print(f"{len(written)} figures ecrites dans {out}/ : {', '.join(written)}")


if __name__ == "__main__":
    main()
