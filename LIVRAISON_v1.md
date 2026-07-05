# Livraison — exploitation des résultats v1 (run `rsna_2026-07-02`)

Ce qui a été ajouté au dépôt, à partir des seuls fichiers déjà commités (aucune valeur retapée à la main).

## 1. Registre d'erreurs (livrable n°1)
- `eval/build_error_register.py` — génère le registre de **30 cas** depuis `baseline_predictions.csv` + `improved_predictions.csv`, sélection déterministe (reproductible).
- `eval/results/rsna_2026-07-02/error_register.csv` — le registre : 1 FN, 3 FP, 15 UA, 4 cas corrects de référence (OK), 7 cas d'effondrement du prompt improved (CM). Colonnes : `case_id, mode, ground_truth, prediction, confidence, error_type, severity, comment, corrective_action, visual_review`.
- `eval/results/rsna_2026-07-02/error_register_synthese.md` — **synthèse §13.5 + top 5 des causes**.

Régénérer : `python eval/build_error_register.py`

⚠️ Deux points à traiter avant de valider le registre :
- **Images** : `data/rsna/` est gitignoré (règles Kaggle). Pour la relecture visuelle case par case (colonne `visual_review = a_faire`), régénérer le dataset avec vos identifiants Kaggle : `scripts/download_rsna.py` puis `scripts/build_rsna_catalogue.py` (seed fixe → mêmes 170 images).
- **Justifications textuelles** : les CSV ne contiennent que classe/confiance. La revue d'hallucinations (HT) demande les justifications complètes, stockées dans la **base SQLite du run Colab**. À récupérer avant expiration de la session, sinon rejouer l'inférence sur les seuls cas du registre.

## 2. Figures de soutenance
- `eval/make_figures.py` — reconstruit 4 figures depuis les fichiers du run, chacune annotée source / date / nombre de cas.
- `eval/results/rsna_2026-07-02/figures/`
  - `fig1_baseline_vs_improved.png` — LE visuel : accuracy, macro-F1, sensibilité, spécificité, taux d'incertitude, accuracy hors uncertain.
  - `fig2_confusion_baseline.png`, `fig3_confusion_improved.png` — matrices mises en forme (comptes + %).
  - `fig4_latences.png` — médiane ~20 s, p95 ~22 s, avec la note « cible < 10 s non atteinte sur T4 ».

Régénérer : `python eval/make_figures.py`

## 3. Dashboard Streamlit
- Nouvelle page **« Résultats d'évaluation »** dans `app/streamlit_app.py` : lit `eval/results/` (métriques, figures, matrices, registre filtrable par type). La démo est autonome, sans dépendre des runs SQLite locaux ni du GPU distant.

Lancer : `streamlit run app/streamlit_app.py` → menu latéral, 3e page.

## 4. Rapport — partie résultats
- `docs/rapport_sections_14_18.md` — sections 14 (protocole), 15 (résultats quantitatifs), 16 (comparaison + diagnostic ancrage few-shot), 17 (analyse qualitative), 18 (registre).

## Articulation avec le re-run 3 prompts
Les formats de fichiers ne changent pas. Quand le prompt v3 et le re-run tombent, il suffit de :
1. déposer les nouveaux CSV/JSON dans un nouveau `eval/results/<date>/` ;
2. relancer `python eval/build_error_register.py --run-dir eval/results/<date>` et `python eval/make_figures.py --run-dir eval/results/<date>`.
La page Streamlit détecte automatiquement le nouveau run dans son menu déroulant.
