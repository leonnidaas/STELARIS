# STELARIS : Pipeline de Classification GNSS & LiDAR

**STELARIS** est un écosystème logiciel conçu pour la classification automatique de l'environnement ferroviaire et la prédiction des erreurs GNSS. Il fusionne des données de positionnement (RINEX), de vérité terrain (Ground Truth) et des nuages de points LiDAR HD de l'IGN pour entraîner des modèles de Machine Learning robustes.

---

## Architecture du Projet

Le projet respecte une séparation stricte entre le code, les données et les modèles :

* **`APPS/`** : Logique métier organisée par modules (Fusion, Labelisation, Entraînement).
* **`DATA/`** : Stockage structuré des données selon leur niveau de maturité.
    * **`00_RAW/`** : Données brutes immuables (IGN, GNSS, GT).
    * **`01_INTERIM/`** : Fichiers synchronisés et extractions LiDAR temporaires.
    * **`02_PROCESSED/`** : Trajets finaux labellisés et prêts à l'usage.
    * **`03_TRAINING/`** : Packs de données (Snapshots) scellés pour l'entraînement.
* **`MODELS/`** : Registre centralisé des modèles entraînés avec leurs métadonnées complètes.

---

## Installation & Configuration

1.  **Environnements** : Le projet utilise des environnements virtuels distincts pour isoler les dépendances.
    * `pip install -r APPS/requirements/ml.txt` pour les modèles.
    * `pip install -r APPS/requirements/rinex.txt` pour le traitement GNSS.
    * `pip install -r APPS/requirements/label.txt` pour la partie LiDAR.
2.  **Configuration** : Les chemins racines et paramètres globaux sont définis dans `APPS/config.yml`.

---

## Pipeline de Travail

### 1. Préparation (Preprocessing)
Le script `preprocessing.py` prépare les données pour les modèles :
* **Segmentation Spatiale** : Découpage des trajets en segments de **2 km** pour éviter l'overfitting géographique.
* **Filtrage Vitesse** : Downsampling automatique des phases d'arrêt (vitesse < 0.5 m/s).
* **Isolation des Séquences** : Les fenêtres temporelles ne chevauchent jamais deux segments différents.
* **Export Artefacts** : Génération d'un pack contenant `.npz` (données), `.pkl` (scaler/encoder) et `.json` (métadonnées).

### 2. Entraînement (Training)
L'orchestrateur `app.py` permet de lancer les deux types de modèles :
* **XGBoost** : Optimisation des hyperparamètres via **Optuna**.
* **GRU (Gated Recurrent Unit)** : Modèle récurrent bidirectionnel avec **Focal Loss** pour gérer le déséquilibre des classes d'environnement.

### 3. Évaluation
`evaluate.py` compare les performances des modèles sur un set de test commun :
* Calcul de la **Balanced Accuracy** et du **F1-Score**.
* Génération de matrices de confusion et courbes ROC/AUC.

---

## Registre des Modèles

Chaque entraînement génère un dossier unique dans `MODELS/` contenant :
* Le fichier de poids (`.keras` pour GRU ou `.json` pour XGBoost).
* Le `scaler.pkl` et le `label_encoder.pkl` utilisés lors de l'entraînement.
* Un `metadata.json` garantissant la traçabilité (features utilisées, ID du dataset source).
<<<<<<< HEAD
=======

>>>>>>> 7739f6a (maj inference)
