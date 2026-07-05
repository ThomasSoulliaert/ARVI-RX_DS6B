# Rapport final — Partie résultats (sections 14 à 18)

> Run de référence : `eval/results/rsna_2026-07-02/` (dataset RSNA Pneumonia Detection Challenge, 170 cas scorés, stratifiés `normal` / `suspected_opacity`, seed fixe). Toutes les valeurs citées proviennent des fichiers commités ; aucune n'est retapée à la main. Figures régénérables par `python eval/make_figures.py`.

## 14. Protocole expérimental

Le périmètre est volontairement restreint à une tâche à trois classes (`normal`, `suspected_opacity`, `uncertain`) sur radiographie thoracique frontale, `uncertain` étant une sortie de sécurité et non une vérité terrain. Le jeu synthétique fourni (30 images) sert uniquement à valider la chaîne logicielle ; l'évaluation de performance porte sur un échantillon réel du RSNA Pneumonia Detection Challenge, converti DICOM → PNG (pixels seuls, aucune métadonnée patient conservée). Le mapping documenté associe `Normal` → `normal`, `Lung Opacity` → `suspected_opacity`, et exclut `No Lung Opacity / Not Normal` du jeu scoré (anomalie visible mais non-opacité : l'inclure fausserait la spécificité). L'échantillonnage est stratifié, reproductible par seed fixe, et les splits sont disjoints. Le dossier `data/rsna/` n'est pas versionné (règles Kaggle) : chaque relecture d'image suppose une régénération locale du catalogue avec des identifiants Kaggle personnels.

Deux prompts sont comparés à modèle et garde-fous identiques : un prompt `baseline` (consigne simple, schéma JSON, trois classes) et un prompt `improved` (procédure pas à pas, exigence d'une observation localisable avant `suspected_opacity`, règle d'incertitude explicite, et trois exemples few-shot de calibration). Le modèle visé est MedGemma en inférence multimodale ; un repli déterministe est prévu si le GPU ou le jeton d'accès sont indisponibles. La couche de garde-fous force `uncertain` en cas de JSON invalide, de qualité d'image insuffisante ou de confiance inférieure à 0,60. Les métriques retenues sont l'accuracy, le macro-F1, la sensibilité sur les opacités, la spécificité sur les normales, le taux de JSON valide, le taux de warning, le taux d'incertitude, la latence (médiane et p95) et une accuracy de couverture calculée hors prédictions `uncertain`, afin de distinguer une vraie erreur d'une prudence assumée. Les hallucinations textuelles sont relevées manuellement à partir des justifications, hors des CSV agrégés.

## 15. Résultats quantitatifs

Sur les 170 cas, la baseline atteint une accuracy de 0,76 et un macro-F1 de 0,57, avec une sensibilité opacités de 0,85 et une spécificité normales de 0,96. Le taux de JSON valide et le taux de warning sont de 100 % sur les deux prompts, ce qui satisfait le socle d'intégration attendu. Le taux d'incertitude baseline est de 21 %, et l'accuracy de couverture (sur les 134 cas non `uncertain`) monte à 0,97 : lorsque la baseline se prononce, elle se trompe rarement, et la majeure partie de son erreur brute est en réalité de la prudence.

La matrice de confusion baseline se lit ainsi : sur 85 normales, 58 sont classées `normal`, 3 en `suspected_opacity` (faux positifs) et 24 en `uncertain` ; sur 85 opacités, 72 sont correctement identifiées, 1 est classée `normal` (faux négatif) et 12 en `uncertain`. Les latences sont élevées et homogènes (médiane 19,9 s, p95 21,9 s), au-dessus de la cible indicative de 10 s : ce point est documenté comme une limite matérielle (GPU T4) et non masqué.

| Métrique | Baseline | Improved |
|---|---|---|
| Accuracy | 0,76 | 0,43 |
| Macro-F1 | 0,57 | 0,30 |
| Sensibilité opacités | 0,85 | 0,00 |
| Spécificité normales | 0,96 | 1,00 |
| Taux d'incertitude | 21 % | 55 % |
| Accuracy hors `uncertain` | 0,97 (n=134) | 0,95 (n=77) |
| JSON valide | 100 % | 100 % |
| Latence médiane | 19,9 s | 21,7 s |

## 16. Comparaison baseline / improved

Contre toute attente, le prompt `improved` dégrade fortement la performance : l'accuracy tombe à 0,43, le macro-F1 à 0,30 et surtout la sensibilité opacités s'effondre à 0,00. Le prompt improved ne produit **aucune** prédiction `suspected_opacity` sur les 170 cas ; sa spécificité de 1,00 est donc trompeuse, car elle traduit une abstention totale sur la classe à risque, pas une meilleure discrimination. Le taux d'incertitude passe de 21 % à 55 %.

Le diagnostic est un **ancrage sur les exemples few-shot** (voir registre d'erreurs, code CM). Le prompt improved fournit trois exemples chiffrés (confiances 0,72 / 0,70 / 0,45). Au lieu de calibrer sa propre confiance sur l'image, le modèle recopie ces valeurs : les 72 vrais positifs de la baseline régressent en `uncertain` avec une confiance quasi constante de 0,40 (dérivée du 0,45 de l'exemple C), donc sous le seuil 0,60 qui déclenche le repli `uncertain` ; en parallèle, quelques opacités passent à `normal` avec une confiance de 0,75 (dérivée du 0,72 de l'exemple A), créant de nouveaux faux négatifs. La régression est massive et systématique : 72 cas sur 72 basculent dans le même sens. C'est une erreur de comportement induite par la formulation du prompt, pas une limite de perception : les mêmes images étaient correctement classées par la baseline.

Cette comparaison est instructive au sens des consignes : une « amélioration » plausible sur le papier peut détruire la métrique clinique la plus importante (la sensibilité aux opacités), et seule une évaluation chiffrée le révèle. Elle motive un prompt v3 dans lequel les exemples ne portent plus de valeurs de confiance imitables (plages qualitatives, ou exemples équilibrés entre classes), suivi d'un re-run comparatif des trois prompts sur le même échantillon.

## 17. Analyse qualitative

Au-delà des chiffres, quatre comportements ressortent. D'abord, la prudence de la baseline est asymétrique et globalement saine : elle préfère `uncertain` à une affirmation risquée, ce qui explique une accuracy de couverture de 0,97. Ensuite, cette prudence a un coût de couverture concentré sur les normales : 24 normales sur 85 (28 %) sont renvoyées `uncertain`, ce qui est le premier poste du taux d'incertitude global et une piste d'amélioration à faible risque. Troisièmement, le seul faux négatif baseline sort à confiance 0,80, au-dessus du seuil de garde-fou : il illustre la limite structurelle d'un filet de sécurité fondé sur un seuil de confiance — une erreur affirmée avec assurance passe au travers. Enfin, les trois faux positifs sortent tous exactement à 0,60, à la frontière de décision, ce qui suggère que la calibration ne sépare pas nettement ces cas d'un vrai positif franc et invite à vérifier les justifications textuelles associées (recherche d'hallucinations, code HT).

La revue des hallucinations textuelles reste à mener sur les justifications complètes : celles-ci ne figurent pas dans les CSV agrégés (classe et confiance seulement) mais dans la base SQLite du run. Cette revue conditionne la requalification éventuelle de certains FP en HT.

## 18. Registre d'erreurs

Le registre (`eval/results/rsna_2026-07-02/error_register.csv`, 30 cas) est construit par script depuis les prédictions commitées, avec sélection déterministe pour être reproductible. Il couvre le faux négatif grave, les trois faux positifs, un échantillon des incertitudes (opacités et normales), quatre cas corrects de référence — pour ne pas ne montrer que des échecs — et sept cas d'effondrement du prompt improved. Chaque ligne porte vérité terrain, prédiction, type (FN / FP / UA / JF / HT, plus CM et OK documentés), sévérité, commentaire analytique et action corrective. La synthèse associée (`error_register_synthese.md`) dégage le top 5 des causes, dominé par l'ancrage few-shot du prompt improved.

Deux réserves de traçabilité accompagnent le registre. La relecture visuelle case par case (colonne `visual_review`) suppose la régénération locale des images RSNA (gitignorées) ; tant qu'elle n'est pas faite, les commentaires restent fondés sur classe, confiance et latence, non sur l'image elle-même. Et la qualification définitive des hallucinations dépend de la base SQLite du run. Ces limites sont explicitées plutôt que dissimulées, conformément à la position du projet : un prototype prudent et défendable, pas une démonstration lissée.
