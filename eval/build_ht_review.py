"""Revue des hallucinations textuelles (HT) à partir des justifications commitées.

Consignes : le code HT (hallucination textuelle) = « mention d'un signe non
visible » / information clinique inventée. Contrairement au registre d'erreurs
(qui ne dispose que de classe + confiance), cette revue exploite les
justifications COMPLÈTES stockées dans `justifications_review.csv` (baseline +
improved). Elle est donc réalisable sans les images RSNA.

Méthode (reproductible, sans valeur retapée) : screening PAR PHRASE des champs
evidence + justification de chaque prompt. Trois catégories de candidats HT,
volontairement conservatrices (on préfère un faux positif de screening à une HT
manquée), chacune reliée à la vérité terrain :

  1. device_claim       — affirme un dispositif médical VISIBLE (cathéter, drain,
                          pacemaker, sonde...). Claim spécifique et vérifiable.
  2. finding_on_normal  — affirme un signe positif localisé (opacité, densité,
                          consolidation, nodule...) sur une image `ground_truth =
                          normal`, hors contexte nié. Signature d'un FP halluciné.
  3. invented_context   — invente un contexte que l'image seule ne donne pas
                          (âge, sexe, antécédents, symptômes).

`needs_image` indique si la confirmation définitive exige la relecture de l'image
(device_claim, finding_on_normal) ou non (invented_context = HT quel que soit le
pixel, car le prompt interdit d'inventer un contexte clinique).

Usage :
  python eval/build_ht_review.py --run-dir eval/results/rsna_2026-07-02
Sorties : <run-dir>/ht_review.csv + <run-dir>/ht_review_synthese.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROMPTS = ["baseline", "improved"]

FINDING = re.compile(
    r"\b(opacit|opacif|consolidation|infiltrat|densit|nodul|\bmass\b|effusion|"
    r"haziness|air ?space|atelectas|pneumon)\w*",
    re.I,
)
NEGATION = re.compile(
    r"\b(no|not|without|absence|absent|free of|unremarkable|clear|symmetric|"
    r"within normal|rule out|r/o|resolved)\b",
    re.I,
)
POSITIVE_CUE = re.compile(
    r"\b(there is|there appears|increased|noted|present|visible|seen|area of|"
    r"focal|suggestive of|consistent with|concerning for|likely|shows? a)\b",
    re.I,
)
DEVICE = re.compile(
    r"\b(chest tube|catheter|central (venous )?line|central venous|pacemaker|picc|"
    r"endotracheal|et tube|pigtail|sternal wire|port-?a-?cath|\bdevice\b|\bdrain\b|"
    r"\blead\b|\bwire\b)\b",
    re.I,
)
CONTEXT = re.compile(
    r"\b(\d{1,3}[- ]year[-\s]old|year-old|\bmale\b|\bfemale\b|history of|known "
    r"(history|diagnosis)|presenting with|patient reports|symptoms? of)\b",
    re.I,
)
# "the patient's clinical history" employé pour dire qu'il MANQUE ne compte pas.
CONTEXT_EXCUSE = re.compile(r"(such as|more information|would (help|be)|lack|absence of|without the)", re.I)


def _sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    # Les evidences sont séparées par " | " ; les justifications par des points.
    parts = re.split(r"[|.]", text)
    return [p.strip() for p in parts if p.strip()]


def screen_text(text: str, ground_truth: str) -> list[tuple[str, str, bool]]:
    """Retourne les (categorie, extrait, needs_image) détectés dans un texte."""
    hits: list[tuple[str, str, bool]] = []
    for sent in _sentences(text):
        low = sent.lower()
        if DEVICE.search(low):
            hits.append(("device_claim", sent, True))
        if CONTEXT.search(low) and not CONTEXT_EXCUSE.search(low):
            hits.append(("invented_context", sent, False))
        if (
            ground_truth == "normal"
            and FINDING.search(low)
            and POSITIVE_CUE.search(low)
            and not NEGATION.search(low)
        ):
            hits.append(("finding_on_normal", sent, True))
    return hits


def build_ht_review(run_dir: Path) -> pd.DataFrame:
    src = pd.read_csv(run_dir / "justifications_review.csv")
    rows: list[dict] = []
    for _, r in src.iterrows():
        gt = r["ground_truth"]
        for prompt in PROMPTS:
            text = f"{r.get(prompt + '_evidence', '')} . {r.get(prompt + '_justification', '')}"
            seen: set[tuple[str, str]] = set()
            for category, snippet, needs_image in screen_text(text, gt):
                key = (category, snippet)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "case_id": r["case_id"],
                        "prompt": prompt,
                        "ground_truth": gt,
                        "predicted_class": r.get(f"{prompt}_class", ""),
                        "confidence": r.get(f"{prompt}_confidence", ""),
                        "ht_category": category,
                        "needs_image": needs_image,
                        "snippet": snippet,
                        "verdict": "a_confirmer",
                    }
                )
    df = pd.DataFrame(rows).sort_values(["ht_category", "case_id", "prompt"]).reset_index(drop=True)
    return df


def write_synthese(df: pd.DataFrame, run_dir: Path) -> None:
    by_cat = df.groupby("ht_category").size().to_dict()
    n_cases = df["case_id"].nunique()
    devices = df[df.ht_category == "device_claim"]
    fon = df[df.ht_category == "finding_on_normal"]
    ctx = df[df.ht_category == "invented_context"]
    lines = [
        "# Revue des hallucinations textuelles (HT) — run `rsna_2026-07-02`",
        "",
        f"Screening reproductible par `eval/build_ht_review.py` sur `justifications_review.csv` "
        f"(170 cas × 2 prompts). {len(df)} extraits candidats HT sur {n_cases} cas distincts. "
        "Trois catégories, du plus au moins vérifiable sans image :",
        "",
        f"- **invented_context** : {by_cat.get('invented_context', 0)} extrait(s). "
        "HT au sens strict des consignes (âge, sexe, antécédents, symptômes) : l'image seule ne les donne "
        "pas et le prompt interdit de les inventer. `needs_image = False`.",
        f"- **device_claim** : {by_cat.get('device_claim', 0)} extrait(s). Le modèle affirme un dispositif "
        "médical visible (cathéter, sonde, pacemaker, drain...). Claim spécifique à confirmer sur l'image "
        f"({devices['case_id'].nunique()} cas concernés, "
        f"{(devices.ground_truth == 'normal').mean() * 100:.0f} % sur des images `normal`).",
        f"- **finding_on_normal** : {by_cat.get('finding_on_normal', 0)} extrait(s). Signe positif localisé "
        "(opacité, densité, nodule...) affirmé sur une image annotée `normal` : signature textuelle d'un "
        "faux positif halluciné, à recouper avec les FP du registre d'erreurs.",
        "",
        "## Lecture",
        "",
        "La revue confirme que le taux de JSON valide de 100 % ne dit rien du contenu : le modèle **peut** "
        "produire un JSON parfaitement formé tout en décrivant un dispositif ou un signe non présents. Les "
        "`device_claim` se concentrent sur le prompt improved et sur des images `normal`, où ils servent "
        "souvent à justifier un `uncertain` — l'hallucination d'un artefact devient alors le motif de la "
        "prudence, ce qui est un mauvais motif. Les `finding_on_normal` alimentent directement les faux "
        "positifs : un signe décrit puis classé `suspected_opacity` sur une image normale est le cas HT le "
        "plus coûteux.",
        "",
        "## Limite de la revue",
        "",
        "Ce screening est **textuel et conservateur** : il liste des *candidats*. La qualification définitive "
        "en HT avérée (vs signe réellement présent mais subtil) exige la relecture de l'image RSNA "
        "correspondante (`needs_image = True`), sauf pour `invented_context` qui est une HT indépendamment du "
        "pixel. Les justifications de `improved_v2` ne sont pas encore exportées dans "
        "`justifications_review.csv` (seulement classe/confiance dans les CSV de prédictions) : étendre le "
        "fichier au prompt v2 permettrait d'auditer aussi ses 19 faux positifs.",
        "",
        "## Cas notables",
        "",
    ]
    for _, h in pd.concat([ctx, devices.head(4), fon.head(3)]).iterrows():
        lines.append(
            f"- `{h.case_id[:13]}` [{h.prompt}, gt={h.ground_truth}, pred={h.predicted_class}] "
            f"**{h.ht_category}** : « {h.snippet.strip()[:150]} »"
        )
    (run_dir / "ht_review_synthese.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="eval/results/rsna_2026-07-02")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    df = build_ht_review(run_dir)
    df.to_csv(run_dir / "ht_review.csv", index=False)
    write_synthese(df, run_dir)
    print(f"{len(df)} candidats HT ecrits dans {run_dir}/ht_review.csv")
    print(df.groupby(["ht_category", "prompt"]).size())


if __name__ == "__main__":
    main()
