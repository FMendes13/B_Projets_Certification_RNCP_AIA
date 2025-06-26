import os
import pickle
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# Fichiers
reference_path = "../data/reference.csv"
model_path = "model.pkl"
previous_model_path = "previous_model.pkl"

# Chargement des données de référence
reference_df = pd.read_csv(reference_path)

# Ajout de la colonne "minute" si manquante
if "minute" not in reference_df.columns and "timestamp" in reference_df.columns:
    reference_df["timestamp"] = pd.to_datetime(reference_df["timestamp"])
    reference_df["minute"] = reference_df["timestamp"].dt.minute

# Features et label
features = ["hour", "day_of_week", "currentSpeed", "freeFlowSpeed", "confidence", "minute"]
label_col = "congestion"

# Mapping texte -> classe
label_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
reference_df[label_col] = reference_df[label_col].str.upper()
y_ref = reference_df[label_col].map(label_mapping)

# Vérification des labels
if y_ref.isna().any():
    print("⚠️ Des valeurs non reconnues sont présentes dans la colonne 'congestion'.")
    print("👉 Lignes concernées :")
    print(reference_df[y_ref.isna()])
    reference_df = reference_df[~y_ref.isna()]
    y_ref = y_ref[~y_ref.isna()]

# Features
X_ref = reference_df[features]

# Chargement du modèle actuel
with open(model_path, "rb") as f:
    new_model = pickle.load(f)

new_preds = new_model.predict(X_ref)
new_score = f1_score(y_ref, new_preds, average="macro")

print(f"📈 Score du nouveau modèle : {new_score:.4f}")

# Comparaison avec un éventuel ancien modèle
if os.path.exists(previous_model_path):
    with open(previous_model_path, "rb") as f:
        old_model = pickle.load(f)

    old_preds = old_model.predict(X_ref)
    old_score = f1_score(y_ref, old_preds, average="macro")
    print(f"📉 Score de l'ancien modèle : {old_score:.4f}")

    if new_score >= old_score:
        print("✅ Nouveau modèle accepté. Mise à jour de la version précédente.")
        os.replace(model_path, previous_model_path)
        with open(model_path, "wb") as f:
            pickle.dump(new_model, f)
    else:
        print("❌ Nouveau modèle rejeté. Moins performant que l'ancien.")
else:
    print("⚠️ Ancien modèle non trouvé. Le nouveau modèle sera accepté.")
    os.replace(model_path, previous_model_path)
    with open(model_path, "wb") as f:
        pickle.dump(new_model, f)
