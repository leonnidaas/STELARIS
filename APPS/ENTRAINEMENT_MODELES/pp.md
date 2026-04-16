# Focus Technique : Pipeline de Prétraitement GNSS (preprocessing.py)

Ce document dissèque la logique de transformation des données brutes en tenseurs d'entraînement. En GNSS ferroviaire, le défi n'est pas la quantité de données, mais leur **corrélation spatiale** et leur **redondance**.

---

## 1. Stratégie de Nettoyage : Downsampling Intelligent
Le script implémente `_downsample_stationary_rows` pour traiter l'immobilite du train.

### Mécanisme
Plutôt qu'un simple filtre de vitesse, le code utilise trois critères combinés :
* **Seuil de vitesse** : `speed_threshold_mps` (souvent 0.5 m/s).
* **Temporisation** : Garde 1 point toutes les $N$ secondes (ex: 30s).
* **Densité** : Garde 1 point toutes les $M$ lignes.

### Justification
En gare, un récepteur GNSS continue de produire des données (bruit de phase, trajets multiples). Si on garde tout, le modèle sur-apprend le bruit spécifique d'une gare. Ce downsampling "dilate" le temps stationnaire pour qu'il ne pèse statistiquement pas plus qu'une section à 300 km/h.

---

## 2. L'Isolation Spatiale : "Grid-Based Grouping"
C'est le cœur de la robustesse du pipeline via `assign_geographic_segments`.

### Fonctionnement
On discrétise la surface terrestre en cellules de taille fixe (ex: 2km x 2km) :
1.  Conversion de la distance cible en degrés de latitude/longitude.
2.  Application d'une division euclidienne (`np.floor`) sur les coordonnées `latitude_gt` et `longitude_gt`.
3.  Génération d'un `segment_id` unique par cellule.



### Pourquoi ce choix ?
Si tu utilises un split aléatoire, le point $t$ (train) et le point $t+1$ (test) sont séparés de quelques mètres. Ils partagent le même environnement (mêmes bâtiments, mêmes arbres). Le modèle "triche" en reconnaissant le voisinage. Le **Grid-Splitting** garantit que si une cellule est en test, le modèle ne l'a jamais vue sous aucun angle pendant l'entraînement.

---

## 3. Construction des Séquences (Tenseurs 3D)
La fonction `create_sequences_centered` prépare les données pour le GRU.

### La règle du "No-Crossing"
Une fenêtre temporelle de taille $L$ ne doit **jamais** chevaucher deux segments géographiques différents.
* **Logique** : Si l'index $i$ appartient au segment A, tous les voisins dans la fenêtre $L$ doivent appartenir au segment A.
* **Gestion des bords** : Si la fenêtre dépasse, on utilise un "padding par répétition" de la valeur la plus proche.

### Justification
Un saut de segment signifie souvent une rupture de continuité physique dans ton dataset (changement de trajet ou saut géographique). Mélanger ces données dans une seule séquence introduirait un gradient artificiel que le GRU interpréterait comme une anomalie de signal.

---

## 4. Le Split Robuste : Stratified Group K-Fold
Le script utilise `_pick_best_stratified_group_split` pour diviser les données.

### Critique de la méthode
1.  **Groupement** : On utilise le `segment_id`. Un segment entier est soit dans le Train, soit dans le Test.
2.  **Stratification** : On s'assure que le ratio des classes (ex: 10% de Tunnels, 20% de Canopée) est identique dans chaque split.
3.  **Optimisation** : Le code itère sur plusieurs splits possibles et choisit celui qui minimise l'écart de distribution par rapport au dataset global.

---

## 5. Synthèse des Artefacts Exportés
Le pipeline ne se contente pas de sortir des fichiers `.npy`, il génère un "bundle" complet :
* **`preprocessed_data.npz`** : Tenseurs compressés (X_train, y_train, etc.).
* **`scaler.pkl`** : L'état du `StandardScaler` (moyenne/écart-type) fitté uniquement sur le train.
* **`metadata.json`** : Le "journal de bord" contenant le nom des features, les ratios de classes et les paramètres de segment.

---

## Critique de l'Expert

* **Point Fort** : La gestion des frontières de segments dans le fenêtrage est exemplaire pour éviter la pollution de données.
* **Point Faible** : La fonction `_sanitize_feature_names` retire les coordonnées brutes. C'est bien pour éviter le sur-apprentissage, mais tu devrais conserver les **coordonnées relatives** (ex: distance parcourue dans le segment) qui peuvent aider le modèle à comprendre la dynamique du train.
* **Performance** : Le calcul du `segment_id` par concaténation de chaînes de caractères (`f"GEO_GRID_{lat}_{lon}"`) est lent. Un hash numérique ou un tuple serait plus efficace pour de très gros volumes de données.


# Rapport d'Analyse : Stratégies d'Entraînement GNSS (GRU vs XGBoost)

Ce document détaille les choix d'architecture et de régularisation pour les modèles de classification GNSS basés sur les scripts `train_gru.py` et `train_xgboost.py`.

---

## 1. Modèle Séquentiel : Bidirectional GRU (`train_gru.py`)

L'utilisation d'un GRU (Gated Recurrent Unit) bidirectionnel est le choix "Deep Learning" pour capturer la dynamique temporelle du signal.

### A. Architecture et Choix Temporels
* **Bidirectionnalité** : 
    > **Justification** : Le signal GNSS à l'instant $t$ est influencé par les masquages passés et futurs (approche hors-ligne). La bidirectionnalité permet au modèle de "voir" l'entrée dans un tunnel avant d'y être et d'en sortir avec le contexte de l'obstruction.
* **Double couche GRU** : La première couche retourne une séquence (`return_sequences=True`) pour que la seconde puisse extraire des abstractions temporelles de plus haut niveau.

### B. Fonction de Perte : Categorical Focal Crossentropy
* **Critique** : C'est le choix le plus robuste de ton code. 
    > **Justification** : En milieu ferroviaire, les classes "Tunnel" ou "Jamming" sont très rares par rapport au "Ciel Ouvert". La *Focal Loss* réduit le poids des exemples faciles (bien classés) pour forcer le modèle à se concentrer sur les cas limites et les classes minoritaires.

### C. Scheduler : CosineDecayRestarts
* **Justification** : Utiliser un cycle de décroissance cosusoïdale avec redémarrages permet de sortir des minima locaux. En GNSS, où le bruit peut créer des paysages de perte complexes, c'est bien plus efficace qu'un simple `ReduceLROnPlateau`.

### D. Aberration Technique : Le mélange TF / Torch
* **Critique** : Tu importes `torch` pour faire un `one_hot` (`to_torch_onehot`) alors que tu utilises `keras` (TensorFlow) pour tout le reste.
    > **Justification du risque** : C'est une hérésie en termes de gestion de mémoire GPU. Charger deux environnements de calcul (LibTorch + TF) peut mener à des `Out Of Memory (OOM)` inutiles. 
    > **Correction** : Utilise `keras.utils.to_categorical` ou `tf.one_hot`.

---

## 2. Modèle Tabulaire : XGBoost & Optuna (`train_xgboost.py`)

XGBoost est ici utilisé comme référence "State-of-the-Art" pour les données tabulaires, optimisé par recherche bayésienne.

### A. Optimisation Hyper-paramétrique (Optuna)
* **Justification** : Contrairement à un GridSearch, Optuna utilise l'élagage (*Pruning*) et des algorithmes TPE pour trouver le meilleur compromis entre profondeur d'arbre et régularisation.
* **StratifiedGroupKFold en CV** : 
    > **Justification** : C'est la suite logique du preprocessing. Si tu ne groupais pas par `segment_id` pendant la validation croisée d'Optuna, tes scores seraient artificiellement gonflés par la fuite de données spatiale.

### B. Pondération des Classes (`sample_weight`)
* **Justification** : XGBoost ne gère pas nativement la *Focal Loss* de la même manière que Keras. L'utilisation de `compute_sample_weight` est le moyen standard d'équilibrer l'importance de chaque échantillon pour le calcul du gradient.

### C. Interprétabilité (Feature Importance)
* **Atout majeur** : Le script exporte l'importance par **Gain** et par **Poids**.
    > **Justification** : En GNSS ferroviaire, le "Gain" est crucial : il indique quelle feature (ex: `CN0_std` ou `NSV`) réduit le plus l'incertitude du modèle. Cela permet de valider physiquement que le modèle n'apprend pas n'importe quoi.

---

## 3. Comparaison Critique des Approches

| Caractéristique | Bidirectional GRU | XGBoost |
| :--- | :--- | :--- |
| **Nature des données** | Tenseurs 3D (Temporel) | Vecteurs plats (Stats) |
| **Sensibilité au bruit** | Élevée (dépend de la fenêtre) | Faible (robuste aux outliers) |
| **Coût Inférence** | Élevé (GPU recommandé) | Très faible (CPU rapide) |
| **Complexité Splitting** | Risque de fuite si $L$ est grand | Simple (point par point) |


---

## 4. Synthèse des Risques Globaux

1.  **Look-ahead Bias** : Dans `train_gru.py`, tu utilises un scaler fitté sur le train pour transformer tout le tenseur de test. C'est correct. Cependant, assure-toi que le `window_size` n'est pas trop grand par rapport à la taille des segments géographiques, sinon le padding répétitif va biaiser les statistiques aux frontières.
2.  **Déséquilibre de Split** :
    ```python
    overlap = set(id_train).intersection(set(id_test))
    ```
    Cette vérification est vitale. Si `overlap > 0`, tes résultats ne valent rien. Ton code lève une erreur, ce qui est une excellente pratique de "Fail-Safe".
3.  **Absence de Post-Processing** : Les deux scripts sortent des prédictions point par point. En ferroviaire, un train ne peut pas être "En Tunnel" pendant 100ms, puis "Ciel Ouvert" pendant 100ms, puis "En Tunnel". 
    > **Justification** : Il manque un filtre de lissage (ex: Filtre de Kalman ou vote majoritaire sur une fenêtre glissante) pour stabiliser les prédictions finales.

---

**Verdict Expert :** Le script **XGBoost** est plus "production-ready" grâce à l'optimisation Optuna et l'analyse d'importance. Le script **GRU** est prometteur pour la recherche de pointe, mais son mélange de frameworks (TF/Torch) et sa dépendance au fenêtrage le rendent plus fragile. 

**Prochaine étape suggérée** : Fusionner les deux approches via un **Ensemble Model** ou utiliser les sorties du GRU comme features pour le XGBoost.