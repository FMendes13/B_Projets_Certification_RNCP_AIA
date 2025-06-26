import pickle
import pandas as pd
from sklearn.metrics import f1_score

# Chargement des données de test
df = pd.read_csv("data/reference.csv")
X = df[["hour", "day_of_week", "currentSpeed", "freeFlowSpeed", "confidence", "minute"]]
y_true = df["label"]

def evaluate(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    return f1_score(y_true, y_pred, average="macro")

if __name__ == "__main__":
    new_score = evaluate("model/traffic_model_v2.pkl")
    current_score = evaluate("model/best_model.pkl")
    if new_score >= current_score:
        # Remplacer le meilleur modèle par le nouveau
        import os
        os.replace("model/traffic_model_v2.pkl", "model/best_model.pkl")
        print("✅ New model deployed.")
    else:
        print("❌ New model rejected.")
