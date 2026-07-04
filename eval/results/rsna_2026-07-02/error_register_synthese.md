# Synthèse du registre d'erreurs (§13.5) — run `rsna_2026-07-02`

Registre : `error_register.csv` (30 cas, générés par `eval/build_error_register.py` depuis les CSV de prédictions du run, sélection déterministe). Composition : 1 FN, 3 FP, 15 UA (6 sur opacités, 9 sur normales), 4 cas de référence corrects (OK), 7 erreurs de comportement modèle (CM) sur le run improved. Le code CM est une extension documentée de la taxonomie FN/FP/UA/JF/HT des consignes : il désigne une erreur imputable au comportement induit par le prompt (et non à la perception de l'image), justifiée ci-dessous. Aucun cas JF n'existe sur ce run (taux de JSON valide = 100 % sur les deux prompts) ; la revue HT (hallucinations textuelles) reste à faire car elle exige les justifications complètes stockées dans la base SQLite du run Colab, non présentes dans les CSV.

## Top 5 des causes d'erreur

1. **Ancrage few-shot du prompt improved (impact maximal).** Le prompt improved ne produit aucune prédiction `suspected_opacity` sur 170 cas : sensibilité opacités 0,85 → 0,00. Les 72 vrais positifs baseline régressent en `uncertain` avec une confiance quasi constante de 0,40 (copie du 0,45 de l'exemple few-shot C), sous le seuil de garde-fou 0,60 qui force `uncertain` ; 3 nouveaux faux négatifs apparaissent à confiance 0,75 (copie du 0,72 de l'exemple A). Le modèle imite les valeurs chiffrées des exemples au lieu de juger l'image. Action : prompt v3 sans confiances imitables dans les exemples (ou exemples équilibrés par classe), puis re-run comparatif des 3 prompts sur les mêmes 170 cas.

2. **Sur-prudence sur les images normales (coût en couverture).** 24 normales sur 85 (28 %) sont renvoyées `uncertain` par la baseline, à confiance 0,30–0,50. C'est le premier poste du taux d'incertitude global (21 %) et la raison pour laquelle la couverture tombe à 134/170 cas. Action : expliciter dans le prompt qu'un champ pulmonaire clair et symétrique justifie `normal` avec confiance ≥ 0,60.

3. **Sous-confiance sur les opacités peu marquées (prudence discutable).** 12 opacités annotées sur 85 (14 %) sont renvoyées `uncertain`. Les cas à confiance 0,50 suggèrent un signal perçu mais non assumé ; la relecture visuelle décidera, cas par cas, entre incertitude acceptable (UA) et occasion manquée.

4. **Faux positifs exactement au seuil de décision.** Les 3 FP baseline sortent tous à confiance 0,60, pile au seuil : la calibration ne distingue pas ces cas d'un vrai positif franc. Hypothèses à confirmer à l'image (superpositions costales, ombres des tissus mous) et dans les justifications SQLite (recherche d'éventuelles HT).

5. **Faux négatif à haute confiance, hors de portée des garde-fous.** Le cas RSNA_bfb32559 (opacité → `normal`, confiance 0,80) est l'erreur la plus grave du run : la confiance élevée le rend invisible pour la règle `conf < 0,60 → uncertain`, et il persiste avec le prompt improved (confiance 0,75). C'est le cas d'école à relire visuellement en priorité et à documenter en soutenance.

## Prérequis avant validation finale du registre

La colonne `visual_review` du registre vaut `a_faire` sur toutes les lignes : les images RSNA ne sont pas dans le dépôt (`data/rsna/` gitignoré, règles Kaggle) et doivent être régénérées localement (`scripts/download_rsna.py` puis `scripts/build_rsna_catalogue.py`, seed fixe → mêmes 170 images). La revue HT exige en plus la base SQLite du run Colab (justifications textuelles complètes) ; si elle n'a pas été récupérée avant expiration de la session, rejouer l'inférence sur les cas du registre uniquement.
