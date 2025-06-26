import pickle
import numpy as np

def test_model_prediction():
    with open("model/traffic_model.pkl", "rb") as f:
        model = pickle.load(f)
    test_input = np.array([[8, 2, 35, 45, 0.8, 15]])
    pred = model.predict(test_input)
    assert pred[0] in [0, 1, 2]
