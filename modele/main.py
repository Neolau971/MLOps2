from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_smote.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print(f"Modèle chargé depuis : {MODEL_PATH}")

def predict_credit(**features):

    # Créer un DataFrame avec une seule ligne
    input_df = pd.DataFrame([features])

    # Prédiction classe
    pred = model.predict(input_df)[0]

    # Probabilité de la classe 1 (défaut)
    proba = float(model.predict_proba(input_df)[0, 1])

    return {
        "prediction": int(pred),
        "probabilite_defaut": round(proba, 4),
    }