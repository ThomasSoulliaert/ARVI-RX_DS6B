# Revue des hallucinations textuelles (HT) — run `rsna_2026-07-02`

Screening reproductible par `eval/build_ht_review.py` sur `justifications_review.csv` (170 cas × 2 prompts). 62 extraits candidats HT sur 25 cas distincts. Trois catégories, du plus au moins vérifiable sans image :

- **invented_context** : 0 extrait(s). HT au sens strict des consignes (âge, sexe, antécédents, symptômes) : l'image seule ne les donne pas et le prompt interdit de les inventer. `needs_image = False`.
- **device_claim** : 26 extrait(s). Le modèle affirme un dispositif médical visible (cathéter, sonde, pacemaker, drain...). Claim spécifique à confirmer sur l'image (13 cas concernés, 35 % sur des images `normal`).
- **finding_on_normal** : 36 extrait(s). Signe positif localisé (opacité, densité, nodule...) affirmé sur une image annotée `normal` : signature textuelle d'un faux positif halluciné, à recouper avec les FP du registre d'erreurs.

## Lecture

La revue confirme que le taux de JSON valide de 100 % ne dit rien du contenu : le modèle **peut** produire un JSON parfaitement formé tout en décrivant un dispositif ou un signe non présents. Les `device_claim` se concentrent sur le prompt improved et sur des images `normal`, où ils servent souvent à justifier un `uncertain` — l'hallucination d'un artefact devient alors le motif de la prudence, ce qui est un mauvais motif. Les `finding_on_normal` alimentent directement les faux positifs : un signe décrit puis classé `suspected_opacity` sur une image normale est le cas HT le plus coûteux.

## Limite de la revue

Ce screening est **textuel et conservateur** : il liste des *candidats*. La qualification définitive en HT avérée (vs signe réellement présent mais subtil) exige la relecture de l'image RSNA correspondante (`needs_image = True`), sauf pour `invented_context` qui est une HT indépendamment du pixel. Les justifications de `improved_v2` ne sont pas encore exportées dans `justifications_review.csv` (seulement classe/confiance dans les CSV de prédictions) : étendre le fichier au prompt v2 permettrait d'auditer aussi ses 19 faux positifs.

## Cas notables

- `RSNA_199d2377` [improved, gt=normal, pred=uncertain] **device_claim** : « There is a device visible in the left upper lung field, possibly a chest tube or other medical device »
- `RSNA_34d37f0b` [improved, gt=suspected_opacity, pred=uncertain] **device_claim** : « The image appears to be a portable chest X-ray with a central line and a possible catheter in the right upper quadrant »
- `RSNA_34d37f0b` [improved, gt=suspected_opacity, pred=uncertain] **device_claim** : « There is a possible opacity in the right lower lung field, but it is difficult to assess due to the projection and potential artifact from the cathete »
- `RSNA_34d37f0b` [improved, gt=suspected_opacity, pred=uncertain] **device_claim** : « The image quality is good, but the projection and potential artifacts from the central line and catheter make it difficult to definitively assess the  »
- `RSNA_199d2377` [baseline, gt=normal, pred=uncertain] **finding_on_normal** : « There appears to be some increased density in the right lower lung field »
- `RSNA_199d2377` [baseline, gt=normal, pred=uncertain] **finding_on_normal** : « The increased density in the right lower lung field could be consistent with pneumonia or other inflammatory processes »
- `RSNA_199d2377` [improved, gt=normal, pred=uncertain] **finding_on_normal** : « Further evaluation is needed to determine the cause of the increased density »
