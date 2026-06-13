# Report d’avancement — forecasting-ml-system

## 1) Ce que j’ai pu analyser (contenu déjà présent)

### Données & preprocessing

- `preprocess_pipeline.py` (script principal)
  - Charge `data/raw/*` (train/stores/oil/holidays/transactions)
  - Merge par `store_nbr` / `date`
  - Nettoyage : suppression de doublons, imputation oil (ffill+bfill), transactions (median par store + global median), remplissage flags holidays
  - Feature engineering basique (temporal + has_promotion + is_holiday)
  - Split chronologique (train/val/test) sans leakage (basé sur des bornes de dates)
  - Sauvegarde en parquet :
    - `data/processed/train.parquet`
    - `data/processed/val.parquet`est c
    - `data/processed/test.parquet`
    - `data/processed/full_featured.parquet`
  - Génère `data/processed/preprocessing_report.txt`

### Bibliothèque “src” (préprocessing plus modulaire)

- `src/data/loader.py`
  - `DataLoader` pour charger les fichiers CSV
  - `merge_data()` pour assembler le panneau store/family/date
- `src/data/preprocess.py`
  - `RetailPreprocessor` (fit/transform)
    - caps quantiles (sales/transactions)
    - fill oil, transactions
    - capping cible `sales`
    - ajout de features temporelles + lags/rolling (lag_1/7/roll_mean_7) et `oil_trend_7d` et `is_payday`
  - Fonctions utilitaires : `merge_sources`, `split_by_time`, `prepare_datasets`
- `src/data/features.py`
  - `FeatureConfig` et pipeline de génération de features (calendrier sin/cos + lags/rolling multi-horizons)
  - `build_features()` (conçu pour éviter le leak)

### Modèles

- `src/models/lstm.py`
  - Pipeline LSTM complet (chargement parquet, scaling StandardScaler, création de séquences par `(store_nbr,family)`, entraînement, prédiction, métriques MAE/RMSE/RMSLE)

### API / tests / autre

- Le repo contient les répertoires :
  - `api/` (FastAPI) + `schemas.py`
  - `tests/` (`test_api.py`, `test_models.py`)
  - `dashboard/` (app)
  - `pipelines/retrain.py`
  - `dvc.yaml`, `docker-compose.yml`

> Note : lors de ma tentative de lecture de plusieurs fichiers (API, metrics/explainability, autoencoder, transformer, training/train.py, tests), l’outil de lecture renvoie du contenu vide. Donc je ne peux pas confirmer la logique exacte de ces fichiers à partir du contenu, même s’ils existent dans l’arborescence.

## 2) Ce qui semble “fait / prêt”

- La partie **data/processed** que tu as travaillée est bien le socle utilisé pour l’entraînement : `preprocess_pipeline.py` produit `data/processed/*` et génère `preprocessing_report.txt`.
- Les datasets `data/processed/train.parquet`, `val.parquet`, `test.parquet` et `full_featured.parquet` semblent donc être la source de vérité côté Kaggle.
- Plusieurs couches “production-style” sont prévues : API, dashboard, retraining pipeline, tests, Docker, DVC.
- Un modèle LSTM avec séquences + scaling + évaluation existe (dans `src/models/lstm.py`).
- Une approche modulaire de preprocessing/features existe aussi dans `src/data/` (fit/transform + génération plus riche), qui peut être soit complémentaire, soit une alternative selon le pipeline réellement utilisé.

## 3) Incohérences / risques techniques relevés

- **Chevauchement preprocessing**
  - Il y a un pipeline complet dans `preprocess_pipeline.py` ET une autre implémentation dans `src/data/preprocess.py` + `src/data/features.py`.
  - Ces pipelines semblent ne pas produire exactement le même type de feature set (ex : `preprocess_pipeline.py` génère 23 features “ciblées” vs `src/data/features.py` génère beaucoup de lags/rolling, sin/cos, etc.).
  - Risque : modèle LSTM s’attend à certaines colonnes (ex `sales`, `transactions`, `dcoilwtico`, features) et peut échouer si un pipeline différent a été utilisé.

- **Fonctionnalités features non alignées**
  - `preprocess_pipeline.py` ne crée pas explicitement `lag_1`, `lag_7`, etc., tandis que `src/data/features.py` et `src/data/preprocess.py` le font.
  - Si `data/processed/*` est généré via `preprocess_pipeline.py`, `src/data/features.py` peut ne jamais être utilisé.
  - Cela peut casser les hypothèses des modèles.

## 4) Ce qui manque probablement (à vérifier/implémenter)

### A) Orchestration training / modèles

- Il semble manquer un **pipeline “unifié”** qui :
  1. exécute le bon feature engineering (choisi)
  2. entraîne tous les modèles (XGB/baseline/LSTM/Transformer/Autoencoder)
  3. sauvegarde les artefacts au même format
  4. enregistre les meilleurs modèles (via MLflow/DVC si prévu)

### B) API de service (contrats + chargement artefacts)

- L’API doit exposer des endpoints cohérents :
  - input schema (format des features / identification store/family / horizon)
  - output schema (forecast + interval éventuel)
  - gestion erreurs + validation
  - chargement de modèle (et versioning)
- Comme je n’ai pas pu lire `api/main.py` / `api/schemas.py` (contenu vide via l’outil), c’est un point à confirmer.

### C) Dashboard

- Le dashboard devrait consommer l’API pour afficher forecasts/anomalies et métriques.
- À confirmer : endpoints utilisés, modèles d’affichage, fréquence refresh.

### D) Évaluation / explainability / anomaly detection

- Le repo contient `src/evaluation/` et `src/models/autoencoder.py`, mais je n’ai pas pu lire leur contenu.
- Il manque probablement :
  - calcul métriques standardisé pour chaque modèle
  - génération d’artefacts explainability (ex: SHAP plots)
  - stratégie seuil anomalies (contamination, percentile, z-score, etc.)

### E) Tests & CI

- Les fichiers de tests existent, mais je n’ai pas pu lire leur contenu.
- À vérifier :
  - tests d’intégration API (statuts/validation)
  - tests de régression modèle (shape, métriques minimales, smoke tests)
  - workflows GitHub/CI (run pytest + lint)

## 5) Statut “avancement” (résumé)

- **Preprocessing** : partiellement prêt (script complet + lib modulaire existante).
- **Feature engineering** : existe (deux implémentations), mais risque d’incohérence d’alignement.
- **Modèles** : LSTM prêt ; autres modèles/déploiement à confirmer.
- **Serving & MLOps** : structure présente (API/dashboard/pipelines/DVC/Docker), mais contenu exact non inspectable via l’outil.
- **Qualité** : tests/metrics/explainability prévus, à confirmer.

## 6) Actions recommandées (priorisées)

1. **Choisir une source unique de feature engineering**
   - Soit tout passe par `preprocess_pipeline.py`, soit tout passe par `src/data/features.py`/`src/data/preprocess.py`.
   - Documenter exactement la “contract schema” des colonnes produites (`data/processed/full_featured.parquet`).

2. **Aligner les hypothèses des modèles**
   - Exemple : `src/models/lstm.py` a besoin d’un ensemble de colonnes features. Vérifier que la génération de `data/processed/*` les contient.

3. **Écrire (ou vérifier) un pipeline training complet**
   - entraîner baseline + XGB + LSTM + Transformer + Autoencoder
   - sauvegarder artefacts sous un format stable (ex: `models/<name>/model.pt` / `model.pkl`)

4. **Vérifier l’API**
   - charger le bon artefact
   - endpoints / schemas conformes
   - tests de smoke API.

5. **Completer évaluation/anomalies**
   - métriques et explainability outputs
   - seuil anomalies reproductible.
