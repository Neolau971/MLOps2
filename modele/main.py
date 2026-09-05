from pathlib import Path

import joblib
import pandas as pd
import gradio as gr
import os


# ============================================================
# Chemins du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "random_forest_smote.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "predictions_credit_scoring.csv"


# ============================================================
# Chargement du modèle
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print(f"Modèle chargé depuis : {MODEL_PATH}")


# ============================================================
# Récupération des colonnes attendues
# ============================================================

def get_expected_features(loaded_model):
    """
    Récupère les variables attendues par le pipeline / modèle.

    Avec un pipeline imblearn, feature_names_in_ est normalement
    disponible sur le pipeline après l'entraînement.
    """

    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    if hasattr(loaded_model, "named_steps"):
        rf_model = loaded_model.named_steps.get("model")

        if rf_model is not None and hasattr(
            rf_model,
            "feature_names_in_",
        ):
            return list(rf_model.feature_names_in_)

    raise AttributeError(
        "Impossible de récupérer les noms des features attendues "
        "depuis le modèle. Sauvegarde explicitement la liste des "
        "colonnes utilisée pendant l'entraînement."
    )


EXPECTED_FEATURES = get_expected_features(model)

print(f"Nombre de features attendues : {len(EXPECTED_FEATURES)}")


# ============================================================
# Prédiction depuis un CSV
# ============================================================

def predict_from_csv(csv_file):
    """
    Reçoit un fichier CSV depuis Gradio, retourne :
    - un aperçu des prédictions ;
    - un message de statut ;
    - le chemin vers un CSV téléchargeable.
    """

    if csv_file is None:
        raise gr.Error("Veuillez importer un fichier CSV.")

    try:
        df_input = pd.read_csv(csv_file)
    except Exception as exc:
        raise gr.Error(
            f"Impossible de lire le CSV : {exc}"
        )

    # Évite une erreur si l'utilisateur envoie aussi TARGET.
    # La cible n'est jamais une feature de prédiction.
    df_input = df_input.drop(
        columns=["TARGET"],
        errors="ignore",
    )

    # Colonnes attendues mais absentes du CSV
    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in df_input.columns
    ]

    # Colonnes présentes mais non utilisées par le modèle
    extra_features = [
        feature
        for feature in df_input.columns
        if feature not in EXPECTED_FEATURES
    ]

    if missing_features:
        preview = ", ".join(missing_features[:15])

        suffix = (
            " ..."
            if len(missing_features) > 15
            else ""
        )

        raise gr.Error(
            f"Le CSV ne contient pas toutes les features attendues. "
            f"Colonnes manquantes ({len(missing_features)}) : "
            f"{preview}{suffix}"
        )

    # Conserver strictement les colonnes attendues,
    # dans l'ordre appris pendant fit().
    X_input = df_input[EXPECTED_FEATURES].copy()

    try:
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]
    except Exception as exc:
        raise gr.Error(
            f"Erreur pendant la prédiction : {exc}"
        )

    # On conserve toutes les colonnes d'origine et on ajoute les sorties
    result_df = df_input.copy()

    result_df["prediction"] = predictions.astype(int)
    result_df["probabilite_defaut"] = probabilities.round(4)

    # Création du dossier artifacts si nécessaire
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarde du CSV de résultats
    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Message de statut affiché dans Gradio
    n_rows = len(result_df)
    n_positive = int((result_df["prediction"] == 1).sum())

    status = (
        f"Prédictions réalisées pour {n_rows} ligne(s). "
        f"Classe 1 prédite pour {n_positive} ligne(s)."
    )

    if extra_features:
        status += (
            f" {len(extra_features)} colonne(s) supplémentaire(s) "
            f"ont été ignorées par le modèle."
        )

    # Tableau affiché, message, fichier téléchargeable
    return result_df.head(100), status, str(OUTPUT_PATH)


# ============================================================
# Interface Gradio
# ============================================================

with gr.Blocks(title="Credit Scoring - Prédictions CSV") as demo:
    gr.Markdown("""
# Credit Scoring — Prédictions depuis un CSV
""")

    csv_input = gr.File(
        label="Fichier CSV à prédire",
        file_types=[".csv"],
        type="filepath",
    )

    predict_button = gr.Button(
        "Lancer les prédictions",
        variant="primary",
    )

    status_output = gr.Textbox(
        label="Statut",
        interactive=False,
    )

    preview_output = gr.Dataframe(
        label="Aperçu des résultats (100 premières lignes)",
        interactive=False,
    )

    download_output = gr.File(
        label="Télécharger le fichier de prédictions",
    )

    predict_button.click(
        fn=predict_from_csv,
        inputs=csv_input,
        outputs=[
            preview_output,
            status_output,
            download_output,
        ],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )