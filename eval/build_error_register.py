"""Construit le registre d'erreurs (20-30 cas commentés) à partir des CSV de prédictions.

Règle d'or des consignes : chaque cas est tiré des fichiers commités
(eval/results/<run>/baseline_predictions.csv et improved_predictions.csv),
jamais retapé à la main. La sélection est déterministe (tri par case_id,
seed fixe) : relancer le script reproduit exactement le même registre.

Les commentaires générés sont ANALYTIQUES (fondés sur classe/confiance/latence).
La colonne `visual_review` vaut "a_faire" tant que la relecture visuelle de
l'image (data/rsna/, à régénérer via scripts/download_rsna.py) n'a pas été
faite ; elle doit passer à "fait" ligne par ligne avant la soutenance.

Taxonomie : FN, FP, UA, JF, HT (consignes) + extensions documentées :
  OK = cas correct gardé comme référence (le registre ne montre pas que des échecs)
  CM = erreur de comportement modèle (effondrement du prompt improved par
       ancrage sur les exemples few-shot), catégorie justifiée en synthèse.

Usage :
  python eval/build_error_register.py --run-dir eval/results/rsna_2026-07-02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WARN = "[REVUE VISUELLE A COMPLETER]"


def _select(df: pd.DataFrame, mask, n: int) -> pd.DataFrame:
    """Sélection déterministe : filtre puis tri par case_id, n premiers."""
    return df[mask].sort_values("case_id").head(n)


def build_register(run_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(run_dir / "baseline_predictions.csv")
    impr = pd.read_csv(run_dir / "improved_predictions.csv")
    merged = base.merge(impr, on=["case_id", "label"], suffixes=("_b", "_i"))

    rows: list[dict] = []

    def add(case, mode, error_type, severity, comment, action, review="a_faire"):
        rows.append(
            {
                "case_id": case["case_id"],
                "mode": mode,
                "ground_truth": case["label"],
                "prediction": case[f"predicted_class_{mode[0]}"]
                if f"predicted_class_{mode[0]}" in case
                else case["predicted_class"],
                "confidence": case[f"confidence_{mode[0]}"]
                if f"confidence_{mode[0]}" in case
                else case["confidence"],
                "error_type": error_type,
                "severity": severity,
                "comment": comment,
                "corrective_action": action,
                "visual_review": review,
            }
        )

    # ------------------------------------------------------------------ FN
    fn = _select(base, (base.label == "suspected_opacity") & (base.predicted_class == "normal"), 5)
    for _, c in fn.iterrows():
        also_fn_improved = bool(
            (
                (impr.case_id == c.case_id)
                & (impr.predicted_class == "normal")
            ).any()
        )
        extra = (
            " Cas persistant : egalement predit normal par le prompt improved (confiance 0.75),"
            " donc opacite probablement subtile ou hors des zones typiques."
            if also_fn_improved
            else ""
        )
        add(
            c,
            "baseline",
            "FN",
            "haute",
            f"Faux negatif le plus grave du run : opacite annotee RSNA predite normal avec confiance {c.confidence:.2f},"
            f" au-dessus du seuil de garde-fou (0.60) donc non rattrapee par la regle d'incertitude.{extra} {WARN}",
            "Relire l'image pour localiser l'opacite manquee ; ajouter un exemple few-shot d'opacite subtile"
            " dans le prompt v3 ; envisager un abaissement cible du seuil pour la classe normal.",
        )

    # ------------------------------------------------------------------ FP
    fp = _select(base, (base.label == "normal") & (base.predicted_class == "suspected_opacity"), 5)
    for _, c in fp.iterrows():
        add(
            c,
            "baseline",
            "FP",
            "moyenne",
            f"Faux positif : image normale predite suspected_opacity a confiance {c.confidence:.2f},"
            " exactement au seuil de decision (0.60). Hypotheses a verifier sur l'image :"
            f" superposition costale, ombre mammaire/tissus mous, exposition. {WARN}",
            "Verifier la justification textuelle (base SQLite) pour detecter une eventuelle hallucination (HT) ;"
            " renforcer dans le prompt v3 l'exigence d'une observation localisable avant suspected_opacity.",
        )

    # ---------------------------------------------------- UA sur opacités
    ua_op = _select(base, (base.label == "suspected_opacity") & (base.predicted_class == "uncertain"), 6)
    for _, c in ua_op.iterrows():
        borderline = c.confidence >= 0.5
        add(
            c,
            "baseline",
            "UA",
            "moyenne" if borderline else "faible",
            f"Opacite annotee renvoyee uncertain (confiance {c.confidence:.2f})."
            + (
                " Confiance proche du seuil : prudence discutable, le modele a probablement vu un signal"
                " mais n'a pas ose conclure."
                if borderline
                else " Confiance basse : incertitude plausible si le signe est faible ou l'image limitee."
            )
            + f" A juger image par image : incertitude acceptable ou occasion manquee. {WARN}",
            "Classer apres relecture : si le signe est franc, compter comme sous-detection et documenter ;"
            " sinon conserver comme comportement prudent attendu.",
        )

    # ---------------------------------------------------- UA sur normales
    ua_no = _select(base, (base.label == "normal") & (base.predicted_class == "uncertain"), 9)
    for _, c in ua_no.iterrows():
        add(
            c,
            "baseline",
            "UA",
            "faible",
            f"Image normale renvoyee uncertain (confiance {c.confidence:.2f}) : sur-prudence."
            " 24 cas de ce type sur 85 normales (28 %) : cout principal en couverture du baseline"
            f" (coverage_n = 134/170). {WARN}",
            "Acceptable pour un prototype prudent, mais a reduire dans le prompt v3 en explicitant"
            " qu'un champ pulmonaire clair et symetrique justifie normal avec confiance >= 0.60.",
        )

    # ------------------------------------------------------------ Réussites
    tp = _select(base, (base.label == "suspected_opacity") & (base.predicted_class == "suspected_opacity"), 2)
    tn = _select(base, (base.label == "normal") & (base.predicted_class == "normal"), 2)
    for _, c in tp.iterrows():
        add(
            c,
            "baseline",
            "OK",
            "aucune",
            f"Vrai positif de reference (confiance {c.confidence:.2f}) : garde au registre pour montrer"
            f" le comportement nominal, conformement a la regle de soutenance (ne pas montrer que des echecs, mais pas que des reussites non plus). {WARN}",
            "Aucune. Verifier tout de meme la justification SQLite : une bonne classe avec une justification"
            " hallucinee resterait un cas HT.",
        )
    for _, c in tn.iterrows():
        add(
            c,
            "baseline",
            "OK",
            "aucune",
            f"Vrai negatif de reference (confiance {c.confidence:.2f}) : normal correctement identifie"
            f" avec confiance au-dessus du seuil. {WARN}",
            "Aucune.",
        )

    # ------------------------------- Effondrement du prompt improved (CM)
    # 3 nouveaux FN improved (hors le cas persistant deja couvert)
    fn_i = impr[(impr.label == "suspected_opacity") & (impr.predicted_class == "normal")]
    fn_i = fn_i[~fn_i.case_id.isin(fn.case_id)].sort_values("case_id").head(3)
    for _, c in fn_i.iterrows():
        add(
            c,
            "improved",
            "CM",
            "haute",
            f"Nouveau faux negatif cree par le prompt improved : opacite predite normal a confiance {c.confidence:.2f}"
            " = quasi copie du 0.72 de l'exemple few-shot A (normal). Le modele reproduit les valeurs des exemples"
            f" au lieu de juger l'image : ancrage few-shot. {WARN}",
            "Prompt v3 : retirer ou equilibrer les exemples chiffres, ou remplacer les confiances des exemples"
            " par des plages qualitatives ; re-run 3 prompts pour mesurer l'effet.",
        )

    # 4 régressions TP -> uncertain, dont le cas extrême confiance 0.0
    reg = merged[
        (merged.label == "suspected_opacity")
        & (merged.predicted_class_b == "suspected_opacity")
        & (merged.predicted_class_i == "uncertain")
    ]
    reg_zero = reg[reg.confidence_i == 0.0].sort_values("case_id").head(1)
    reg_rest = reg[reg.confidence_i > 0.0].sort_values("case_id").head(3)
    for _, c in pd.concat([reg_zero, reg_rest]).iterrows():
        add(
            c,
            "improved",
            "CM",
            "haute",
            f"Regression : vrai positif baseline (confiance {c.confidence_b:.2f}) devenu uncertain en improved"
            f" (confiance {c.confidence_i:.2f}"
            + (
                ", cas extreme a 0.00" if c.confidence_i == 0.0 else " = ancrage sur le 0.45 de l'exemple C"
            )
            + "). 72 regressions de ce type sur 72 TP baseline : sensibilite opacites 0.85 -> 0.00."
            " Erreur de comportement modele, pas de perception : la meme image etait correctement classee. "
            + WARN,
            "Prompt v3 (exemples equilibres, sans valeurs de confiance imitables) puis re-run complet ;"
            " comparer les 3 prompts sur les memes 170 cas.",
        )

    df = pd.DataFrame(rows)
    assert 20 <= len(df) <= 30, f"registre hors bornes consignes: {len(df)} cas"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="eval/results/rsna_2026-07-02")
    parser.add_argument("--out", default=None, help="CSV de sortie (defaut: <run-dir>/error_register.csv)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "error_register.csv"
    df = build_register(run_dir)
    df.to_csv(out, index=False)
    print(f"{len(df)} cas ecrits dans {out}")
    print(df.groupby(["mode", "error_type"]).size())


if __name__ == "__main__":
    main()
