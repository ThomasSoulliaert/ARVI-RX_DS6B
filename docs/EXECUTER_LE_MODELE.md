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

Routes : `GET /` (health check), `POST /predict` (champs `file` + `mode`),
`POST /warmup` (charge le modèle en arrière-plan, répond immédiatement),
`GET /ready` (état du chargement : `loaded` / `loading` / `error`).

---

## 4. Recommandation d'architecture

Le modèle s'exécute **dans le process qui appelle `run_prediction`**. Pour éviter
de bloquer vos machines (pas de GPU) :

> **Faites tourner l'API FastAPI là où il y a un GPU** (Colab/serveur), et faites
> appeler cette API par l'interface web. Le front reste léger et ne dépend ni de
> `torch`, ni du token HF, ni du GPU.

---

## 5. Exécuter l'API sur Google Colab (GPU gratuit) + tunnel

Pas de serveur GPU dédié ? On fait tourner **cette même API FastAPI sur Colab** et on
l'expose via un tunnel public gratuit (cloudflared). L'interface Streamlit locale
appelle alors l'URL du tunnel.

### Côté Colab (Exécution → Modifier le type d'exécution → **T4 GPU**)

```python
# Cellule 1 — cloner le dépôt et installer les dépendances
!git clone https://github.com/ThomasSoulliaert/ARVI-RX_DS6B.git
%cd ARVI-RX_DS6B
!pip install -q -r requirements.txt

# Cellule 2 — authentification Hugging Face (licence MedGemma déjà acceptée, cf. §2.b)
from huggingface_hub import login
login()  # coller le token hf_...

# Cellule 3 — lancer l'API en arrière-plan
import subprocess
api = subprocess.Popen(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

# Cellule 4 — ouvrir le tunnel public (l'URL https://xxxx.trycloudflare.com s'affiche)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8000 --no-autoupdate
```

### Côté machine locale

Lancer Streamlit normalement, puis coller l'URL `https://xxxx.trycloudflare.com`
dans le champ **« URL API distante »** de la barre latérale (ou définir la variable
d'environnement `ARVI_API_URL` avant de lancer l'app). L'inférence part alors sur
le GPU Colab ; le run est aussi loggé dans la base SQLite locale pour le dashboard.

Limites à connaître :

- l'URL trycloudflare **change à chaque session** Colab → la re-coller à chaque fois ;
- Colab gratuit coupe la session après quelques heures d'inactivité ;
- le premier chargement du modèle prend 2 à 5 min (téléchargement des poids) :
  dès que l'URL est collée dans Streamlit, l'interface appelle `/warmup`
  automatiquement, affiche « Chargement du modèle… » et **désactive le bouton
  Analyser** jusqu'à ce que le modèle soit prêt — aucun timeout n'est montré
  pendant cette phase. Les appels suivants prennent ~15-20 s sur T4 ;
- le tunnel gratuit coupe toute requête > ~100 s (erreur 524) : c'est pour cela
  que le chargement passe par `/warmup` (non bloquant) et jamais par un premier
  `/predict` ;
- pour l'**évaluation batch** (RSNA, métriques), pas besoin de tunnel : lancer
  `python eval/run_evaluation.py --mode rsna` directement dans Colab et télécharger
  les CSV produits dans `eval/outputs/`.

## 6. Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `RuntimeError: ... modèle *gated* ...` | Pas authentifié sur HF | Voir §2.b |
| Warning « Aucun GPU CUDA détecté » | Pas de GPU | Normal, ça tourne en CPU (lent). Voir §2.c / §4 |
| `FileNotFoundError` sur un prompt | (corrigé) chemin résolu depuis la racine du dépôt | Mettre à jour `src/inference.py` |
| Inférence très lente | Exécution CPU | Utiliser une machine GPU (§4) |
