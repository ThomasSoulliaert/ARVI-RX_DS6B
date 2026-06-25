# Assistant radiologue virtuel responsable

Prototype pédagogique d'assistant radiologue virtuel pour radiographie thoracique frontale.

Ce dépôt n'est pas un dispositif médical. La sortie ne doit jamais être utilisée pour diagnostiquer, trier ou orienter un patient.

> Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.

## Objectif

L'application permet de tester une chaîne complète et traçable :

```text
Image uploadée -> prétraitement -> inférence -> garde-fous -> JSON -> SQLite -> dashboard
```

Les sorties autorisées sont strictement limitées à :

- `normal`
- `suspected_opacity`
- `uncertain`

Le modèle actuellement branché est un modèle jouet déterministe. Il sert à valider l'intégration web/API/SQLite/dashboard. Un vrai modèle pré-entraîné, par exemple MedGemma ou un modèle Hugging Face compatible, pourra être branché plus tard dans `src/inference.py`.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell :

```powershell
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

Linux/macOS :

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

## Lancer Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'interface contient deux pages :

- `Analyse image` : upload, affichage de l'image, bouton d'analyse, résultat expérimental et JSON complet.
- `Historique / Dashboard` : statistiques et dernières prédictions sauvegardées en SQLite.

Pour tester rapidement, utiliser par exemple :

```text
data/sample_images/CXR_SYN_002_suspected_opacity.png
```

## Lancer l'API FastAPI

```bash
uvicorn api.main:app --reload
```

Endpoints disponibles :

```text
GET /
POST /predict
```

Exemple avec `curl` :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png" \
  -F "mode=improved"
```

Le paramètre `mode` accepte :

- `baseline`
- `improved`

La réponse contient un JSON structuré avec au minimum :

```json
{
  "image_quality": "good|medium|poor",
  "predicted_class": "normal|suspected_opacity|uncertain",
  "confidence": 0.0,
  "visual_elements": [],
  "justification": "",
  "limitations": [],
  "warning": "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
  "model_name": "",
  "prompt_version": "",
  "latency_ms": 0
}
```

## Pipeline d'intégration

Le point d'entrée central est :

```python
from src.pipeline import run_prediction
```

```python
run_prediction(
    image_path="data/sample_images/CXR_SYN_002_suspected_opacity.png",
    mode="improved",
    save=True,
)
```

Le pipeline réalise :

1. validation du fichier image ;
2. ouverture avec Pillow ;
3. calcul qualité image : largeur, hauteur, luminosité, contraste ;
4. appel du modèle via `predict_with_model(...)` ;
5. normalisation et garde-fous JSON ;
6. règle d'incertitude si confiance faible ou qualité mauvaise ;
7. sauvegarde SQLite si `save=True` ;
8. retour du dictionnaire JSON final.

## SQLite et dashboard

Les prédictions web/API sont sauvegardées localement dans :

```text
data/predictions.sqlite
```

Ce fichier est généré à l'exécution et ne doit pas être commité.

La table `runs` stocke notamment :

- `case_id`
- `image_path`
- `mode`
- `model_name`
- `prompt_version`
- `predicted_class`
- `confidence`
- `image_quality`
- `prediction_json`
- `latency_ms`
- `created_at`

Le dashboard Streamlit lit cette base avec :

- `fetch_recent_runs(...)`
- `summarize_runs(...)`

## Tests

Vérification rapide :

```bash
python -m compileall -q src api app eval finetuning tests
python -m pytest -q
```

Le dépôt contient aussi une évaluation jouet :

```bash
python eval/run_evaluation.py --mode toy
```

## Organisation

```text
ARVI-RX_DS6B/
├── app/           # Streamlit / Gradio
├── api/           # FastAPI
├── data/          # images synthétiques de test
├── docs/          # documentation projet
├── eval/          # évaluation jouet
├── finetuning/    # essais futurs
├── prompts/       # prompts baseline/improved et schéma JSON
├── sql/           # schéma SQLite
├── src/           # pipeline, inférence, prétraitement, garde-fous, SQLite
└── tests/         # tests smoke et intégration
```

## Statut du modèle

Le modèle actuel est un placeholder pédagogique :

```text
predict_with_model(...) -> toy_predict(...)
```

Il ne constitue pas une inférence médicale réelle. Il permet seulement de tester l'intégration.

La future intégration ML devra se faire dans :

```text
src/inference.py
```

L'objectif est de remplacer progressivement l'intérieur de `predict_with_model(...)` ou de le faire appeler une fonction VLM telle que `vlm_predict(...)`, sans modifier Streamlit, FastAPI ou `src/pipeline.py`.

## Références possibles

Les extensions avancées doivent rester expérimentales, traçables et justifiées.

- MedGemma : modèle médical image-texte sous conditions d'usage.
- Gemma / Unsloth : fine-tuning expérimental LoRA/QLoRA.
- MIMIC-CXR / MIMIC-CXR-JPG : données avec accès contrôlé.
- CheXpert : jeu de données de radiographies thoraciques.

Chaque source externe doit être citée avec version, licence ou conditions d'accès.

## Points de vigilance

- Ne jamais présenter la sortie comme un diagnostic.
- Ne pas supprimer la classe `uncertain`.
- Ne pas commiter de données patient réelles.
- Ne pas commiter `data/predictions.sqlite`.
- Toujours conserver le warning obligatoire dans l'interface et les sorties JSON.
