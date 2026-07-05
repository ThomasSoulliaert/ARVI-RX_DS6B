# Rapport final — Assistant radiologue virtuel responsable (ARVI-RX_DS6B)

> Prototype pédagogique d'IA multimodale pour l'analyse prudente de radiographies thoraciques frontales.
> **Non destiné au diagnostic. Validation par un professionnel qualifié requise.**

Ce document est le rapport unique de soutenance. Les sections 1 à 13 posent le cadre, l'architecture, les données, les prompts, les garde-fous et l'éthique ; les sections 14 à 18 (partie résultats) sont développées en détail dans [`rapport_sections_14_18.md`](rapport_sections_14_18.md) et synthétisées ici. Les documents de référence complémentaires sont [`architecture.md`](architecture.md), [`evaluation_protocol.md`](evaluation_protocol.md), [`ethique_et_limites.md`](ethique_et_limites.md) et [`licences_et_sources.md`](licences_et_sources.md).

## Sommaire

1. Contexte et positionnement non-clinique
2. Périmètre et contrat du projet
3. Architecture logicielle et pipeline
4. Données : jeu synthétique et échantillon RSNA
5. Contrat de sortie JSON
6. Prompts : baseline, improved, improved_v2
7. Garde-fous et règle d'incertitude
8. Intégration applicative (web, API, SQLite)
9. Modèle d'inférence (MedGemma) et repli
10. Protocole et métriques d'évaluation
11. Reproductibilité et qualité
12. Éthique, limites et risques
13. Licences et sources externes
14–18. Résultats (protocole, quantitatif, comparaison, qualitatif + HT, registre)
19. Conclusion et perspectives

---

## 1. Contexte et positionnement non-clinique

Les modèles multimodaux produisent un texte médical fluide qui ne garantit pas la correction clinique. Le projet ne vise pas le diagnostic mais l'apprentissage d'une **démarche d'ingénierie responsable** : périmètre restreint, baseline, métriques, logs, garde-fous, démo web et limites documentées. Le dépôt n'est pas un dispositif médical ; aucune sortie ne doit servir à diagnostiquer, trier ou orienter un patient. L'avertissement obligatoire figure dans l'interface, la sortie JSON, le README, le rapport et la soutenance.

## 2. Périmètre et contrat du projet

| Élément | Cadrage |
|---|---|
| Entrée | Une radiographie thoracique frontale |
| Sorties | `normal`, `suspected_opacity`, `uncertain` |
| Preuve minimale | JSON valide, warning, logs, métriques, cas d'erreur |
| Données | Synthétiques ou publiques, autorisées et dé-identifiées |
| Finalité | Prototype éducatif data/IA, pas une aide au diagnostic |

La tâche est volontairement réduite à trois classes ; `uncertain` est une sortie de sécurité et non une vérité terrain assignable à un dataset.

## 3. Architecture logicielle et pipeline

Chaîne complète et traçable (détail dans [`architecture.md`](architecture.md)) :

```text
Image upload → prétraitement → VLM / toy model → garde-fous → JSON → UI → logs SQLite
```

| Composant | Rôle |
|---|---|
| `src/preprocessing.py` | validation fichier, chargement image, resize, contrôle qualité |
| `src/inference.py` | inférence MedGemma (`vlm_predict`) ou repli jouet déterministe |
| `src/guardrails.py` | validation JSON, warning, règle d'incertitude |
| `src/metrics.py` | accuracy, macro-F1, sensibilité, spécificité, validité JSON |
| `src/database.py` | init SQLite et stockage des runs |
| `api/main.py` | endpoint FastAPI `/predict` (+ `/warmup`, `/ready`) |
| `app/streamlit_app.py` | interface upload + dashboard + page « Résultats d'évaluation » |
| `eval/` | évaluation, figures, registre d'erreurs, revue HT |

Point de jonction clé : l'application appelle `run_prediction` → `predict_with_model`, qui tente le vrai VLM et retombe sur le jouet sans casser la chaîne logicielle.

## 4. Données : jeu synthétique et échantillon RSNA

Deux niveaux, avec des rôles distincts (détail dans [`data/README.md`](../data/README.md) et [`evaluation_protocol.md`](evaluation_protocol.md)) :

- **Jeu synthétique** (`data/synthetic_cases.csv`, `data/sample_images/`) : images jouet imitant grossièrement une radio, pour valider le **code** (flux, logs, JSON, garde-fous). Un score parfait dessus n'est **pas** une performance médicale.
- **Échantillon RSNA** (RSNA Pneumonia Detection Challenge) : 170 vraies radios (20 `smoke` + 150 `dev`), converties DICOM → PNG (pixels seuls), stratifiées, seed fixe. Mapping documenté : `Normal → normal`, `Lung Opacity → suspected_opacity`, `No Lung Opacity / Not Normal` **exclu** du jeu scoré. `data/rsna/` est gitignoré (règles Kaggle, non redistribuable) : chacun régénère le catalogue localement.

## 5. Contrat de sortie JSON

Toute prédiction respecte un schéma fixe ([`prompts/json_schema.md`](../prompts/json_schema.md)) :

```json
{
  "image_quality": "good | limited | poor",
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": [],
  "justification": "",
  "limitations": [],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
  "model_name": "", "prompt_version": "", "latency_ms": 0
}
```

Le taux de JSON valide est de 100 % sur les trois prompts, mais cette mesure est faite après les garde-fous, qui normalisent la sortie : elle est donc toujours conforme par construction. Mesuré sur la sortie brute du modèle, le taux réel est de 100 % pour baseline et improved_v2, et de 99,4 % pour improved (une seule sortie mal formée sur 170).

## 6. Prompts : baseline, improved, improved_v2

Le cœur de l'« amélioration mesurée » repose sur trois prompts, à modèle et garde-fous **identiques** ([`prompts/`](../prompts/)) :

- **`baseline`** : consigne simple, schéma JSON, trois classes.
- **`improved`** : procédure pas à pas, exigence d'une observation localisable avant `suspected_opacity`, règle d'incertitude, **et trois exemples few-shot avec des confiances chiffrées** (0,72 / 0,70 / 0,45).
- **`improved_v2`** : même procédure, mais les exemples **ne portent plus de valeurs de confiance imitables** et une consigne explicite interdit de recopier les nombres des exemples.

Ce triptyque permet une comparaison contrôlée : le seul facteur qui change entre `improved` et `improved_v2` est le traitement des confiances dans les exemples. Le résultat (§16) est le finding central du projet.

## 7. Garde-fous et règle d'incertitude

`src/guardrails.py` applique, après inférence : validation du schéma JSON, présence du warning, et **repli `uncertain`** si JSON invalide, qualité d'image insuffisante ou confiance `< 0,60`. `uncertain` n'est jamais supprimé de l'espace de sortie. Limite structurelle identifiée (§17) : un filet fondé sur un seuil de confiance ne peut pas rattraper une erreur affirmée avec une confiance élevée.

## 8. Intégration applicative (web, API, SQLite)

- **Streamlit** (`app/streamlit_app.py`) : page *Analyse image* (upload, JSON, warning), page *Historique / Dashboard* (SQLite), page *Résultats d'évaluation* (métriques, figures, matrices, registre, **revue HT**) — démo autonome, sans dépendre du GPU distant.
- **FastAPI** (`api/main.py`) : `POST /predict` (multipart), plus `/warmup` et `/ready` pour charger MedGemma en arrière-plan.
- **SQLite** : chaque run (case_id, mode, modèle, prompt_version, classe, confiance, qualité, JSON, latence, timestamp) est journalisé ; `data/predictions.sqlite` n'est pas commité.

## 9. Modèle d'inférence (MedGemma) et repli

L'inférence réelle est assurée par `src/inference.py` sur le modèle vision-langage `google/medgemma-4b-it`. Le modèle est chargé une seule fois puis mis en cache, avec un préchargement en arrière-plan pour ne pas bloquer la première requête, et le bloc JSON de la réponse est extrait même lorsque le modèle ajoute du texte autour.

MedGemma étant *gated* (licence à accepter et jeton d'accès requis), il n'est pas toujours disponible. Dans ce cas, `predict_with_model` bascule sur un modèle jouet déterministe pour vérifier que la chaîne logicielle fonctionne, sans prétendre à une analyse médicale. Ce repli est interdit en évaluation RSNA : une garde dédiée interrompt alors le run, pour ne jamais présenter des métriques du modèle jouet comme des résultats réels. Les conditions d'accès et licences sont détaillées en section 13.

## 10. Protocole et métriques d'évaluation

Détail dans [`evaluation_protocol.md`](evaluation_protocol.md). Jeux : `smoke` (20), `dev` (150), `final` (registre 20–30 cas). Métriques : accuracy, macro-F1, sensibilité opacités, spécificité normales, taux de JSON valide, taux de warning, taux d'incertitude, latence médiane/p95, et **accuracy de couverture** (hors `uncertain`) pour distinguer une vraie erreur d'une prudence assumée. Taxonomie d'erreurs : FN, FP, UA, JF, HT (consignes) + OK (référence) et CM (comportement induit par le prompt), documentés.

## 11. Reproductibilité et qualité

- **Smoke test / CI** (`.github/workflows/ci.yml`) : `pytest`, `compileall`, évaluation jouet à chaque push.
- **Chaîne de résultats reproductible, sans valeur retapée** : `eval/run_evaluation.py` (prédictions + métriques) → `eval/consolidate_summary.py` (résumé 3 prompts) → `eval/make_figures.py` (figures) → `eval/build_error_register.py` (registre) → `eval/build_ht_review.py` (revue HT). Chaque figure porte sa source, sa date et son n.

## 12. Éthique, limites et risques

Détail dans [`ethique_et_limites.md`](ethique_et_limites.md). Ligne rouge : aucun diagnostic, aucun tri, aucun remplacement d'un professionnel. Aucune donnée patient réelle commitée. Risques principaux : hallucination textuelle (mention d'un signe/dispositif non visible — quantifiée §17), confiance non calibrée, sensibilité au prompt (démontrée §16), sous-représentativité de l'échantillon. Règle de soutenance : ne jamais montrer que des réussites.

## 13. Licences et sources externes

Voir [`licences_et_sources.md`](licences_et_sources.md) : code MIT ; MedGemma *gated* (Health AI Developer Foundations, non redistribué, non-dispositif médical) ; RSNA à accès Kaggle contrôlé, non redistribuable, dé-identifié (pixels seuls) ; licences des bibliothèques.

---

## 14–18. Résultats (synthèse)

Développement complet : [`rapport_sections_14_18.md`](rapport_sections_14_18.md). Run de référence `eval/results/rsna_2026-07-02/`, 170 cas RSNA, trois prompts.

| Métrique | baseline | improved | improved_v2 |
|---|---|---|---|
| Accuracy | 0,76 | 0,43 | 0,86 |
| Macro-F1 | 0,57 | 0,30 | 0,58 |
| Sensibilité opacités | 0,85 | 0,00 | 0,95 |
| Spécificité normales | 0,96 | 1,00 | 0,78 |
| Taux d'incertitude | 21 % | 55 % | 0 % |
| Accuracy hors `uncertain` | 0,97 (n=134) | 0,95 (n=77) | 0,86 (n=170) |

Le macro-F1 reste bas (~0,57) car la classe `uncertain` n'a pas de vérité terrain dans le dataset : son F1 vaut 0 par construction et tire la moyenne vers le bas, alors que les F1 des deux classes réelles sont autour de 0,85.

**Le finding central.** Le prompt `improved` s'effondre (sensibilité opacités 0,85 → 0,00) par **ancrage sur les confiances des exemples few-shot** : le modèle recopie les nombres au lieu de juger l'image, et 72 vrais positifs baseline régressent en `uncertain`. `improved_v2`, qui retire ces confiances imitables, **récupère 81 opacités** et restaure la sensibilité à 0,95 — preuve directe que la régression venait du prompt, pas de la perception.

**Les coûts assumés.** `improved_v2` gagne en sensibilité mais (a) fait chuter la spécificité de 0,96 à 0,78 (19 nouveaux FP) et (b) ne produit **plus jamais `uncertain`**, sortant ses 4 faux négatifs à confiance 0,95 — **hors de portée du garde-fou** `conf < 0,60`. Supprimer la classe de sécurité retire le filet.

**Système retenu.** Nous retenons `baseline` comme système de référence : il est prudent, bien calibré (couverture 0,97) et conserve l'abstention `uncertain`, que les consignes posent en garde-fou à ne pas sacrifier à l'accuracy. `improved_v2` n'est pas le système livré mais la preuve qu'un prompt corrigé restaure la sensibilité ; un futur réglage devra combiner cette détection avec l'abstention de la baseline.

**Traçabilité des erreurs.** Registre de 28 cas couvrant les trois prompts (`error_register.csv`) ; revue HT (`ht_review.csv`) : 0 contexte inventé, 26 dispositifs affirmés visibles, 36 signes décrits sur des images `normal` (signature des FP). Réserves explicites : relecture visuelle des images RSNA à faire (gitignorées), revue HT à étendre à `improved_v2`.

## 19. Conclusion et perspectives

Le prototype démontre une **méthode**, pas un modèle spectaculaire : chaîne complète et traçable, sorties JSON et warnings systématiquement conformes, et surtout une **amélioration mesurée honnêtement** — y compris son échec intermédiaire (`improved`) et les compromis de sa correction (`improved_v2`). Perspectives, par priorité :

1. **Calibration de la confiance** et réintroduction d'une bande d'incertitude dans `improved_v2` pour restaurer le garde-fou sans revenir à la sur-prudence.
2. **Relecture visuelle** du registre et **extension de la revue HT à `improved_v2`** (export des justifications v2).
3. **Réduction de latence** (< 10 s) : quantification, modèle plus léger ou batching.
4. **Pistes COULD** : localisation visuelle (heatmap), ablation systématique de prompts, fine-tuning LoRA/QLoRA — à n'engager qu'après ces consolidations, et à documenter comme dépendances réglementées.
