import numpy as np
import joblib
import pandas as pd

# 1. Chargement des fichiers
data = np.load('/media/leon_peltzer/DATA/leon/STELARIS/DATA/03_TRAINING/2026-04-09_16-38/2026-04-09_16-38_preprocessed_data.npz', allow_pickle=True)
le = joblib.load('/media/leon_peltzer/DATA/leon/STELARIS/DATA/03_TRAINING/2026-04-09_16-38/2026-04-09_16-38_label_encoder.pkl')

# 2. Extraction
X = data['X_train']  # On prend les données scalées
y = data['y_train']
features = [
    "gnss_feat_NSV", "gnss_feat_EL mean", "gnss_feat_EL std", 
    "gnss_feat_pdop", "gnss_feat_CN0 std", "gnss_feat_CMC_l1"
]

# 3. Identification de l'index du label 'signal denied'
if "signal_denied" in le.classes_:
    class_idx = np.where(le.classes_ == "signal_denied")[0][0]
    
    # Masque pour isoler les échantillons concernés
    mask = (y == class_idx)
    X_denied = X[mask]
    
    print(f"Nombre d'échantillons 'signal_denied' trouvés : {len(X_denied)}")
    
    if len(X_denied) > 0:
        # Création d'un DataFrame pour une analyse statistique rapide
        df_denied = pd.DataFrame(X_denied, columns=features)
        print("\nStatistiques des features pour 'signal denied' (Valeurs Scalées) :")
        print(df_denied.describe())
        
        # Regarder les 5 premières lignes brutes
        print("\nExemple de valeurs (5 premières lignes) :")
        print(df_denied.head())
    else:
        print("Aucun échantillon trouvé avec ce label.")
else:
    print(f"Le label 'signal denied' n'est pas dans l'encodeur. Classes : {le.classes_}")