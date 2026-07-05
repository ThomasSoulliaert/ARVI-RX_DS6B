# Licences et sources externes
> **Auteur du cadrage :** Badr TAJINI — Solution Delivery, filière Data — 2025-2026
> **Rédaction dépôt :** équipe ARVI-RX_DS6B

Cette page répond à l'exigence explicite de l'appel d'offre : *« indiquer la source, la version, la licence ou les conditions d'accès, les restrictions de redistribution, les traitements d'anonymisation et les limites d'interprétation »* pour toute dépendance réglementée (modèle, dataset) et documenter les licences des bibliothèques. Elle centralise ce qui était jusqu'ici épars dans le dépôt.

> ⚠️ Les chaînes de licence ci-dessous sont indiquées de bonne foi à la date de rédaction. Conformément à l'esprit du projet (dépendances externes traitées comme réglementées), **elles doivent être revérifiées à la source avant toute soutenance ou publication** : les conditions d'un modèle *gated* ou d'un dataset à accès contrôlé peuvent changer.

## 1. Code du dépôt

| Élément | Licence | Notes |
|---|---|---|
| Code ARVI-RX_DS6B (ce dépôt) | MIT (`LICENSE`) | Code pédagogique. Aucune garantie, aucun usage clinique. |

## 2. Modèle d'inférence — MedGemma

| Champ | Valeur |
|---|---|
| Ressource | `google/medgemma-4b-it` (variante instruction-tuned, multimodale image-texte) |
| Source | Model card Hugging Face : https://huggingface.co/google/medgemma-4b-it |
| Version utilisée | `medgemma-4b-it` (branchée dans `src/inference.py`, constante `MODEL_ID`) |
| Conditions d'accès | Modèle **gated** : acceptation de la licence sur Hugging Face + jeton d'accès requis (`HF_TOKEN` / `huggingface-cli login`). Régi par les *Health AI Developer Foundations Terms of Use* de Google. |
| Redistribution | Les poids ne sont **pas** redistribués dans ce dépôt ; ils sont téléchargés à l'exécution sur une machine authentifiée. Ne jamais committer les poids. |
| Limites d'usage | Google positionne MedGemma comme un outil de **recherche / développement**, non comme un dispositif médical, non validé cliniquement. Cohérent avec la ligne rouge du projet (voir `docs/ethique_et_limites.md`). |
| Repli | Si le modèle est indisponible (pas de GPU/token), `predict_with_model` retombe sur un modèle jouet déterministe pour ne pas casser la chaîne logicielle ; la garde anti-fallback de `eval/run_evaluation.py` interdit ce repli en évaluation RSNA. |

## 3. Jeu de données réel — RSNA Pneumonia Detection Challenge

| Champ | Valeur |
|---|---|
| Ressource | RSNA Pneumonia Detection Challenge (radiographies thoraciques + labels d'opacité) |
| Source | Kaggle : https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge — images dérivées du NIH ChestX-ray, annotations RSNA / Society of Thoracic Radiology |
| Version / sous-ensemble | 170 cas scorés (20 `smoke` + 150 `dev`), stratifiés `normal` / `suspected_opacity`, seed fixe ; catalogue régénéré par `scripts/download_rsna.py` + `scripts/build_rsna_catalogue.py` |
| Conditions d'accès | Compte Kaggle + acceptation des règles de la compétition. Accès contrôlé. |
| Redistribution | **Interdite** : les images ne sont **pas** versionnées (`data/rsna/` gitignoré). Chaque utilisateur régénère le catalogue localement avec ses propres identifiants. |
| Anonymisation | Conversion DICOM → PNG en **pixels seuls** ; aucune métadonnée patient (nom, date, identifiant, centre) n'est lue ni conservée. Le mapping de labels et les exclusions (`No Lung Opacity / Not Normal`) sont documentés dans `docs/evaluation_protocol.md`. |
| Limites d'interprétation | Labels d'opacité à visée « pneumonia challenge », non équivalents à un diagnostic radiologique complet ; l'échantillon de 170 cas n'est pas représentatif d'une population clinique. |

## 4. Autres datasets cités (non utilisés en l'état)

Mentionnés par l'appel d'offre comme pistes possibles, **non intégrés** au dépôt. À traiter avec les mêmes exigences s'ils sont mobilisés :

| Ressource | Accès | Point de vigilance |
|---|---|---|
| MIMIC-CXR / MIMIC-CXR-JPG | PhysioNet, accès contrôlé (formation CITI + DUA) | Non redistribuable, dé-identification stricte imposée |
| CheXpert | Stanford AIMI, accès sur demande | Conditions d'usage propres, citer la version |

## 5. Bibliothèques Python (licences)

Licences des principales dépendances (`requirements.txt`). Toutes permissives, compatibles avec un usage pédagogique ; à revérifier à la version épinglée au moment du rendu.

| Bibliothèque | Licence usuelle |
|---|---|
| transformers, accelerate | Apache-2.0 |
| torch, torchvision | BSD-3-Clause |
| streamlit, gradio | Apache-2.0 |
| fastapi, pydantic, python-multipart | MIT / Apache-2.0 |
| uvicorn, httpx | BSD-3-Clause |
| pandas, numpy, scikit-learn | BSD-3-Clause |
| matplotlib | Matplotlib License (style BSD/PSF) |
| pillow | HPND (permissive) |
| opencv-python | Apache-2.0 |
| pydicom | MIT |
| requests | Apache-2.0 |
| pytest | MIT |

## 6. Rappel de conformité

- Aucun fichier patient réel, même pseudonymisé, n'est ajouté au dépôt.
- Les sorties du prototype ne sont pas des diagnostics et portent toujours l'avertissement obligatoire.
- Toute extension (fine-tuning LoRA/QLoRA, MedGemma adapté, dataset réel supplémentaire) doit compléter ce tableau **avant** expérimentation, avec source, version, licence, conditions d'accès et limites.
