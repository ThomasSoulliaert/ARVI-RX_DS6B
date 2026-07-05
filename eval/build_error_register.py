"""Construit le registre d'erreurs (20-30 cas commentés) à partir des CSV de prédictions.

Règle d'or des consignes : chaque cas est tiré des fichiers commités
(eval/results/<run>/{baseline,improved,improved_v2}_predictions.csv), jamais
retapé à la main. La sélection est déterministe (tri par case_id) : relancer le
script reproduit exactement le même registre.

Le registre couvre les TROIS prompts pour rendre la comparaison lisible :
  - baseline : le comportement prudent de référence (FN grave, FP au seuil, UA) ;
  - improved : l'effondrement par ancrage few-shot (CM) ;
  - improved_v2 : la correction — sensibilité restaurée (OK récupérés) au prix de
    la spécificité (nouveaux FP) et de faux négatifs à HAUTE confiance (0.95, donc
    hors de portée du garde-fou conf < 0.60), le prompt v2 ne produisant plus jamais
    `uncertain`.

Les commentaires générés sont ANALYTIQUES (fondés sur classe/confiance/latence,
et sur le comportement croisé des trois prompts). La colonne `visual_review` vaut
"a_faire" tant que la relecture visuelle de l'image (data/rsna/, à régénérer via
scripts/download_rsna.py) n'a pas été faite ; elle doit passer à "fait" ligne par
ligne avant la soutenance.

Taxonomie : FN, FP, UA, JF, HT (consignes) + extensions documentées :
  OK = cas correct gardé comme référence (le registre ne montre pas que des échecs)
  CM = erreur de comportement modèle (effondrement du prompt improved par ancrage
       sur les exemples few-shot), catégorie justifiée en synthèse.

Usage :
  python eval/build_error_register.py --run-dir eval/results/rsna_2026-07-02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WARN = "[REVUE VISUELLE A COMPLETER]"


def _load(run_dir: Path) -> pd.DataFrame:
    """Fusionne les prédictions des trois prompts avec des suffixes explicites.

    Colonnes de sortie : case_id, label, pred_b/conf_b, pred_i/conf_i, pred_v/conf_v.
    improved_v2 est optionnel : s'il manque, ses colonnes valent NA (rétro-compat).
    """
    def one(name: str, suffix: str) -> pd.DataFrame:
        df = pd.read_csv(run_dir / f"{name}_predictions.csv")
        return df.rename(columns={"predicted_class": f"pred{suffix}", "confidence": f"conf{suffix}"})[
            ["case_id", "label", f"pred{suffix}", f"conf{suffix}"]
        ]

    merged = one("baseline", "_b").merge(one("improved", "_i"), on=["case_id", "label"])
    v2_path = run_dir / "improved_v2_predictions.csv"
    if v2_path.exists():
        merged = merged.merge(one("improved_v2", "_v"), on=["case_id", "label"])
    else:
        merged["pred_v"] = pd.NA
        merged["conf_v"] = pd.NA
    return merged


def _pick(df: pd.DataFrame, mask, n: int) -> pd.DataFrame:
    """Sélection déterministe : filtre puis tri par case_id, n premiers."""
    return df[mask].sort_values("case_id").head(n)


def build_register(run_dir: Path) -> pd.DataFrame:
    m = _load(run_dir)
    rows: list[dict] = []

    def add(case, mode, pred, conf, error_type, severity, comment, action, review="a_faire"):
        rows.append(
            {
                "case_id": case["case_id"],
                "mode": mode,
                "ground_truth": case["label"],
                "prediction": pred,
                "confidence": conf,
                "error_type": error_type,
                "severity": severity,
                "comment": comment,
                "corrective_action": action,
                "visual_review": review,
            }
        )

    # =================================================================== baseline
    # ------------------------------------------------------------------ FN grave
    fn = _pick(m, (m.label == "suspected_opacity") & (m.pred_b == "normal"), 1)
    for _, c in fn.iterrows():
        add(
            c, "baseline", c.pred_b, c.conf_b, "FN", "haute",
            f"Faux negatif le plus grave du run : opacite annotee RSNA predite normal a confiance {c.conf_b:.2f},"
            " au-dessus du seuil de garde-fou (0.60) donc non rattrapee. Cas PERSISTANT sur les trois prompts :"
            f" improved -> {c.pred_i}, improved_v2 -> {c.pred_v} a confiance {float(c.conf_v):.2f} (encore plus"
            f" assure). Aucun prompt ne le rattrape : c'est LE cas a relire en priorite. {WARN}",
            "Relire l'image pour localiser l'opacite manquee ; l'ajouter comme exemple few-shot d'opacite subtile ;"
            " un seuil de confiance ne peut structurellement pas attraper une erreur affirmee a 0.95.",
        )

    # ------------------------------------------------------------------ FP au seuil
    fp = _pick(m, (m.label == "normal") & (m.pred_b == "suspected_opacity"), 3)
    for _, c in fp.iterrows():
        add(
            c, "baseline", c.pred_b, c.conf_b, "FP", "moyenne",
            f"Faux positif : image normale predite suspected_opacity a confiance {c.conf_b:.2f},"
            " exactement au seuil de decision (0.60). Hypotheses a verifier sur l'image :"
            f" superposition costale, ombre mammaire/tissus mous, exposition. {WARN}",
            "Verifier la justification textuelle (revue HT, eval/build_ht_review.py) pour detecter une"
            " eventuelle hallucination ; exiger une observation localisable avant suspected_opacity.",
        )

    # ------------------------------------------------------ UA sur opacités (baseline)
    ua_op = _pick(m, (m.label == "suspected_opacity") & (m.pred_b == "uncertain"), 3)
    for _, c in ua_op.iterrows():
        borderline = c.conf_b >= 0.5
        add(
            c, "baseline", c.pred_b, c.conf_b, "UA", "moyenne" if borderline else "faible",
            f"Opacite annotee renvoyee uncertain (confiance {c.conf_b:.2f})."
            + (
                " Confiance proche du seuil : prudence discutable, signal probablement vu mais non assume."
                if borderline
                else " Confiance basse : incertitude plausible si le signe est faible ou l'image limitee."
            )
            + f" improved_v2 sur ce cas -> {c.pred_v}. A juger image par image. {WARN}",
            "Classer apres relecture : si le signe est franc, compter comme sous-detection ;"
            " sinon conserver comme comportement prudent attendu.",
        )

    # ------------------------------------------------------ UA sur normales (baseline)
    ua_no = _pick(m, (m.label == "normal") & (m.pred_b == "uncertain"), 3)
    for _, c in ua_no.iterrows():
        add(
            c, "baseline", c.pred_b, c.conf_b, "UA", "faible",
            f"Image normale renvoyee uncertain (confiance {c.conf_b:.2f}) : sur-prudence."
            " 24 normales sur 85 (28 %) dans ce cas : premier poste du taux d'incertitude baseline (21 %)"
            f" et cout principal en couverture (coverage_n = 134/170). {WARN}",
            "Acceptable pour un prototype prudent ; improved_v2 supprime cette sur-prudence"
            " (taux d'incertitude 0 %) mais au prix d'une chute de specificite : compromis a assumer.",
        )

    # ------------------------------------------------------------- Réussites baseline
    tp = _pick(m, (m.label == "suspected_opacity") & (m.pred_b == "suspected_opacity"), 2)
    tn = _pick(m, (m.label == "normal") & (m.pred_b == "normal"), 2)
    for _, c in tp.iterrows():
        add(
            c, "baseline", c.pred_b, c.conf_b, "OK", "aucune",
            f"Vrai positif de reference (confiance {c.conf_b:.2f}) : comportement nominal, garde au registre"
            f" pour ne pas montrer que des echecs (regle de soutenance). {WARN}",
            "Aucune. Verifier tout de meme la justification (revue HT) : bonne classe + justification hallucinee"
            " resterait un cas HT.",
        )
    for _, c in tn.iterrows():
        add(
            c, "baseline", c.pred_b, c.conf_b, "OK", "aucune",
            f"Vrai negatif de reference (confiance {c.conf_b:.2f}) : normal correctement identifie"
            f" au-dessus du seuil. {WARN}",
            "Aucune.",
        )

    # =================================================================== improved (CM)
    # Nouveaux FN créés par improved (opacité -> normal), hors le FN baseline déjà couvert.
    fn_i = _pick(
        m,
        (m.label == "suspected_opacity") & (m.pred_i == "normal") & (~m.case_id.isin(fn.case_id)),
        2,
    )
    for _, c in fn_i.iterrows():
        add(
            c, "improved", c.pred_i, c.conf_i, "CM", "haute",
            f"Nouveau faux negatif cree par improved : opacite predite normal a confiance {c.conf_i:.2f}"
            " = quasi copie du 0.72 de l'exemple few-shot A (normal). Le modele reproduit les valeurs des"
            f" exemples au lieu de juger l'image : ancrage few-shot. baseline sur ce cas -> {c.pred_b}. {WARN}",
            "Corrige par improved_v2 (exemples sans confiances imitables) ; conserve ici pour documenter le"
            " mecanisme de la regression.",
        )

    # Régressions TP baseline -> uncertain improved, dont le cas extrême confiance 0.0.
    reg_mask = (
        (m.label == "suspected_opacity") & (m.pred_b == "suspected_opacity") & (m.pred_i == "uncertain")
    )
    reg_zero = _pick(m, reg_mask & (m.conf_i == 0.0), 1)
    reg_rest = _pick(m, reg_mask & (m.conf_i > 0.0), 1)
    for _, c in pd.concat([reg_zero, reg_rest]).iterrows():
        add(
            c, "improved", c.pred_i, c.conf_i, "CM", "haute",
            f"Regression : vrai positif baseline (confiance {c.conf_b:.2f}) devenu uncertain en improved"
            f" (confiance {c.conf_i:.2f}"
            + (", cas extreme a 0.00" if c.conf_i == 0.0 else " = ancrage sur le 0.45 de l'exemple C")
            + "). 72 regressions de ce type sur 72 TP baseline : sensibilite opacites 0.85 -> 0.00."
            f" improved_v2 recupere ce cas -> {c.pred_v}. {WARN}",
            "Corrige par improved_v2 ; illustre qu'une 'amelioration' plausible peut detruire la metrique"
            " clinique la plus importante, seule l'evaluation chiffree le revele.",
        )

    # =============================================================== improved_v2
    if m["pred_v"].notna().any():
        # Nouveaux FP : cout de la specificite (0.96 -> 0.78), a HAUTE confiance.
        fp_v = _pick(m, (m.label == "normal") & (m.pred_v == "suspected_opacity"), 5)
        for _, c in fp_v.iterrows():
            add(
                c, "improved_v2", c.pred_v, c.conf_v, "FP", "moyenne",
                f"Faux positif introduit par improved_v2 : image normale predite suspected_opacity a confiance"
                f" {float(c.conf_v):.2f} (elevee). baseline sur ce cas -> {c.pred_b}. En poussant le modele a"
                " conclure (fin de la sur-prudence), le prompt v2 sur-appelle l'opacite : 19 FP sur 85 normales,"
                f" specificite 0.96 -> 0.78. {WARN}",
                "Compromis a assumer en soutenance : sensibilite +0.10 contre specificite -0.18. Verifier via la"
                " revue HT si ces FP s'accompagnent d'une observation inventee ; sinon ajuster le seuil de decision.",
            )

        # Nouveaux FN v2 à HAUTE confiance (0.95) : le garde-fou ne peut pas les attraper.
        fn_v = _pick(
            m,
            (m.label == "suspected_opacity") & (m.pred_v == "normal") & (~m.case_id.isin(fn.case_id)),
            3,
        )
        for _, c in fn_v.iterrows():
            add(
                c, "improved_v2", c.pred_v, c.conf_v, "FN", "haute",
                f"Faux negatif improved_v2 a confiance {float(c.conf_v):.2f} : opacite predite normal. baseline le"
                f" renvoyait prudemment uncertain ({c.pred_b}) ; v2 le transforme en erreur AFFIRMEE. Comme v2 ne"
                " produit jamais uncertain et sort ses confiances a 0.70/0.80/0.95, le garde-fou conf < 0.60 ne se"
                f" declenche jamais : le filet de securite est court-circuite. {WARN}",
                "Point de vigilance majeur : documenter que gagner en accuracy en supprimant `uncertain` retire le"
                " garde-fou. Envisager de reintroduire une bande d'incertitude ou une calibration de confiance.",
            )

        # OK récupérés : opacités que improved envoyait en uncertain, corrigées par v2.
        rec = _pick(
            m,
            (m.label == "suspected_opacity") & (m.pred_v == "suspected_opacity") & (m.pred_i == "uncertain"),
            2,
        )
        for _, c in rec.iterrows():
            add(
                c, "improved_v2", c.pred_v, c.conf_v, "OK", "aucune",
                f"Opacite RECUPEREE par improved_v2 (confiance {float(c.conf_v):.2f}) : improved l'avait perdue en"
                " uncertain (ancrage few-shot), v2 la reclasse correctement suspected_opacity. 81 opacites"
                f" recuperees de la sorte : c'est le moteur du gain de sensibilite 0.00 -> 0.95. {WARN}",
                "Aucune. Cas de reference pour montrer que la correction du prompt fonctionne.",
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
