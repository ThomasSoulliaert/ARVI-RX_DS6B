# Faire tourner le modèle IA (MedGemma) — guide équipe web

Ce guide explique comment exécuter le modèle derrière l'interface web. **Vous n'avez
pas besoin de toucher au code ML** : un seul point d'entrée fait tout le travail.

> ⚠️ Prototype pédagogique, **non destiné au diagnostic**.

---

## 1. Le seul appel dont vous avez besoin

```python
from src.pipeline import run_prediction

result = run_prediction("chemin/vers/image.png", mode="improved")  # ou "baseline"
```

`run_prediction` enchaîne tout automatiquement : preprocessing → modèle MedGemma
(`src/inference.py`) → garde-fous de sécurité → log SQLite. Vous récupérez un
**dict JSON prêt à afficher** :

```json
{
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "visual_evidence": ["..."],
  "justification": "...",
  "limitations": ["..."],
  "warning": "Prototype pédagogique. Non destiné au diagnostic...",
  "model_name": "google/medgemma-4b-it",
  "latency_ms": 1234,
  "case_id": "...",
  "mode": "improved"
}
```

L'API FastAPI (`api/main.py`) et les apps Streamlit/Gradio (`app/`) sont déjà
branchées dessus. Le plus simple côté front : **appeler l'API HTTP** plutôt que
d'exécuter le modèle dans votre process.

---

## 2. Prérequis pour exécuter le modèle

### a. Dépendances

```bash
pip install -r requirements.txt
```

### b. Accès au modèle Hugging Face (OBLIGATOIRE)

`google/medgemma-4b-it` est un modèle **gated** : sans authentification, le
chargement échoue avec un message d'erreur explicite. Étapes (une seule fois) :

1. Accepter la licence : <https://huggingface.co/google/medgemma-4b-it>
2. Créer un token : <https://huggingface.co/settings/tokens>
3. S'authentifier :
   ```bash
   huggingface-cli login
   # ou bien définir la variable d'environnement :
   export HF_TOKEN=hf_xxxxxxxx        # Windows PowerShell : $env:HF_TOKEN="hf_xxxxxxxx"
   ```

Au premier lancement, ~8 Go de poids sont téléchargés et mis en cache.

### c. Matériel

- **Avec GPU CUDA** (Colab, serveur) : chargement en `bfloat16`, rapide. ✅
- **Sans GPU** : le modèle tourne **quand même** sur CPU (`float32`). Un *warning*
  s'affiche au chargement — ce n'est **pas** une erreur. C'est juste **lent**
  (plusieurs minutes par image) et gourmand en RAM. Acceptable pour un test,
  pas pour une démo fluide.

---

## 3. Lancer l'API

```bash
uvicorn api.main:app --reload
```

Test :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png" \
  -F "mode=improved"
```

Routes : `GET /` (health check), `POST /predict` (champs `file` + `mode`).

---

## 4. Recommandation d'architecture

Le modèle s'exécute **dans le process qui appelle `run_prediction`**. Pour éviter
de bloquer vos machines (pas de GPU) :

> **Faites tourner l'API FastAPI là où il y a un GPU** (Colab/serveur), et faites
> appeler cette API par l'interface web. Le front reste léger et ne dépend ni de
> `torch`, ni du token HF, ni du GPU.

---

## 5. Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `RuntimeError: ... modèle *gated* ...` | Pas authentifié sur HF | Voir §2.b |
| Warning « Aucun GPU CUDA détecté » | Pas de GPU | Normal, ça tourne en CPU (lent). Voir §2.c / §4 |
| `FileNotFoundError` sur un prompt | (corrigé) chemin résolu depuis la racine du dépôt | Mettre à jour `src/inference.py` |
| Inférence très lente | Exécution CPU | Utiliser une machine GPU (§4) |
