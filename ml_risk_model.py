import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "risk_model.pkl"


def train_demo_model():
    data = [
        ("chest pain shortness of breath diabetes high blood pressure", "High"),
        ("chest pain breathing hypertension", "High"),
        ("severe chest pain dizziness sweating", "High"),
        ("shortness of breath diabetes", "High"),
        ("chest pressure arm pain", "High"),

        ("diabetes fatigue frequent urination", "Medium"),
        ("headache dizziness", "Medium"),
        ("stomach pain vomiting", "Medium"),
        ("cough fever body pain", "Medium"),
        ("breathing cough fever", "Medium"),

        ("skin rash itching", "Low"),
        ("mild fever cold", "Low"),
        ("sore throat runny nose", "Low"),
        ("minor headache", "Low"),
        ("small skin itching", "Low"),
    ]

    X = [item[0] for item in data]
    y = [item[1] for item in data]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_demo_model()

    return joblib.load(MODEL_PATH)


def predict_ml_risk(symptoms, medical_history):
    model = load_model()

    text = " ".join(symptoms + medical_history).lower()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    prob_dict = {
        classes[i]: round(float(probabilities[i]), 3)
        for i in range(len(classes))
    }

    confidence = max(prob_dict.values())
    uncertainty = round(1 - confidence, 3)

    return {
        "ml_risk_prediction": prediction,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "probabilities": prob_dict,
        "input_text": text
    }