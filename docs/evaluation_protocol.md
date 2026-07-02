# Protocole d'évaluation
> **Author :** Badr TAJINI 
> **Solution Delivery - filière Data** 
>  **Année académique :** 2025-2026
## Jeux de cas

- `smoke` : 20 images pour vérifier la chaîne.
- `dev` : 100 à 150 cas si un vrai dataset est utilisé.
- `final` : 20 à 30 cas commentés pour la soutenance.

Le jeu synthétique fourni sert uniquement à valider le pipeline logiciel : chargement, inférence jouet, JSON, logs, métriques et garde-fous. Un score parfait sur ce jeu ne constitue pas une performance médicale.

### Dataset réel : RSNA Pneumonia Detection Challenge

`scripts/download_rsna.py` + `scripts/build_rsna_catalogue.py` construisent un catalogue
`data/rsna/rsna_catalogue.csv` au même format que `data/synthetic_cases.csv`, à partir de
vraies radios (converties DICOM -> PNG, pixels uniquement, aucune métadonnée patient
utilisée). `data/rsna/` n'est jamais commit (gitignoré) ; chaque étudiant régénère le
catalogue localement avec ses propres credentials Kaggle.

**Mapping des labels** (décision documentée) :

| Classe RSNA (`stage_2_detailed_class_info.csv`) | Label ARVI |
|---|---|
| `Normal` | `normal` |
| `Lung Opacity` | `suspected_opacity` |
| `No Lung Opacity / Not Normal` | **exclu** du catalogue scoré |

`No Lung Opacity / Not Normal` désigne des radios avec une anomalie visible mais qui
n'est pas une opacité pulmonaire (ex. cardiomégalie, épanchement...). Elles ne sont ni
`normal` ni `suspected_opacity` au sens de cette tâche à 3 classes, et `uncertain` est une
sortie de sécurité du modèle, pas une vérité terrain qu'on peut assigner à un dataset :
les inclure sous `normal` fausserait la spécificité mesurée. Elles sont donc exclues du
catalogue scoré et écrites à part dans `data/rsna/rsna_excluded_not_normal.csv` pour une
relecture qualitative ultérieure (ex. vérifier que le modèle n'y déclenche pas de faux
`suspected_opacity`), sans compter dans le macro-F1.

L'échantillonnage est stratifié (équilibré `normal` / `suspected_opacity`), reproductible
(seed fixe), et les splits `smoke` (20 cas) / `dev` (150 cas) sont disjoints.

`eval/run_evaluation.py --mode rsna` utilise `predict_with_model` (tentative d'inférence
MedGemma réelle, repli automatique sur le modèle jouet déterministe si GPU/token
indisponibles) au lieu de `toy_predict`. Les métriques ajoutent une vue "coverage" :
`coverage_accuracy` = exactitude calculée uniquement sur les prédictions où le modèle ne
répond pas `uncertain`, pour distinguer une vraie erreur d'une prudence acceptable.

## Métriques minimales

- Accuracy.
- Macro-F1.
- Sensibilité sur les cas `suspected_opacity`.
- Spécificité sur les cas `normal`.
- Taux de JSON valide.
- Taux de warning présent.
- Taux d'incertitude.
- Hallucinations textuelles détectées manuellement.
- Latence médiane.

## Taxonomie d'erreurs

| Code | Signification | Exemple |
|---|---|---|
| FN | Faux négatif | anomalie présente prédite normale |
| FP | Faux positif | image normale prédite suspecte |
| UA | Incertitude acceptable | signes faibles ou image limitée |
| JF | JSON format error | sortie non exploitable |
| HT | Hallucination textuelle | mention d'un signe non visible |

## Règle de soutenance

Ne jamais montrer seulement des réussites. Une bonne défense montre aussi les faux positifs, les faux négatifs, les incertitudes et les limites de qualité image.

## Smoke test attendu

Avant toute démonstration, le dépôt doit passer un contrôle court :

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python eval/run_evaluation.py --mode toy --out-dir /tmp/assistant-radio-eval --db-path /tmp/assistant-radio-evidence.sqlite
```

Ce test ne remplace pas l'analyse d'erreurs. Il vérifie seulement que le dépôt est exécutable, que les avertissements sont présents et que les sorties restent structurées.
