# test_model.py
import pickle
import numpy as np

def test_prediction():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = np.array([[8, 2, 45.0, 55.0, 0.9, 30]])  # 6 features
    pred = model.predict(sample)

    assert pred[0] in [0, 1, 2], "La prédiction n'est pas valide."

if __name__ == "__main__":
    test_prediction()
    print("✅ test_model.py : le test passe.")
