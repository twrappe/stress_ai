import joblib
import numpy as np

def test_model_loads():
    model = joblib.load("models/stress_classifier.pkl")
    assert model is not None

def test_model_predicts():
    model = joblib.load("models/stress_classifier.pkl")
    sample = np.array([[0.6, 0.9, 0.3, 0.6, 0.2, 0.5]])
    prediction = model.predict(sample)
    assert prediction[0] in ['stressed', 'calm']
