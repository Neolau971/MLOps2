from pathlib import Path
import hashlib
import logging
import time
import uuid
import numpy as np
from database import init_database, log_prediction, log_feature_statistics
import joblib
import pandas as pd
import gradio as gr
import os
from monitoring import build_feature_statistics
from datetime import datetime, timezone
import json

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

MODEL_NAME = "random_forest_smote"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
DECISION_THRESHOLD = float(
    os.getenv("DECISION_THRESHOLD", "0.5")
)

MONITORING_DIR = PROJECT_ROOT / "monitoring_data"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

MAX_PRODUCTION_SAMPLE_ROWS = 1_000

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


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
def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def get_input_summary(df_input: pd.DataFrame) -> dict:
    """
    Ne stocke pas les lignes complètes.
    Conserve uniquement des statistiques agrégées.
    """

    numeric_df = df_input.select_dtypes(
        include=["number"]
    )

    summary = {
        "missing_values_total": int(df_input.isna().sum().sum()),
        "missing_values_ratio": float(
            df_input.isna().mean().mean()
        ),
        "numeric_columns": int(numeric_df.shape[1]),
    }

    return summary


def build_event(
    request_id: str,
    status: str,
    latency_ms: float,
    latency_per_row_ms: float | None = None,
    input_missing_ratio: float | None = None,
    output_positive_rate: float | None = None,
    df_input: pd.DataFrame | None = None,
    predictions: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    error: Exception | None = None,
) -> dict:
    """
    Construit un événement à insérer dans prediction_logs.
    """

    if df_input is not None:
        schema_string = "|".join(
            df_input.columns.astype(str)
        )

        input_schema_hash = sha256_text(
            schema_string
        )

        input_rows = int(len(df_input))
        input_columns = int(df_input.shape[1])

        input_summary = get_input_summary(
            df_input
        )

    else:
        input_schema_hash = None
        input_rows = None
        input_columns = None
        input_summary = None

    if predictions is not None:
        prediction_positive_count = int(
            (predictions == 1).sum()
        )

        prediction_negative_count = int(
            (predictions == 0).sum()
        )

    else:
        prediction_positive_count = None
        prediction_negative_count = None

    if probabilities is not None and len(probabilities) > 0:
        probability_mean = float(
            np.mean(probabilities)
        )

        probability_min = float(
            np.min(probabilities)
        )

        probability_max = float(
            np.max(probabilities)
        )

    else:
        probability_mean = None
        probability_min = None
        probability_max = None

    return {
        "request_id": request_id,
        "event_type": "prediction",
        "status": status,

        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_sha": None,

        "input_rows": input_rows,
        "input_columns": input_columns,
        "input_schema_hash": input_schema_hash,
        "input_file_hash": None,

        "prediction_positive_count": prediction_positive_count,
        "prediction_negative_count": prediction_negative_count,
        "probability_mean": probability_mean,
        "probability_min": probability_min,
        "probability_max": probability_max,
        "decision_threshold": DECISION_THRESHOLD,

        "latency_ms": float(latency_ms),
        "latency_per_row_ms": (
            float(latency_per_row_ms)
            if latency_per_row_ms is not None
            else None
        ),
        "input_missing_ratio": (
            float(input_missing_ratio)
            if input_missing_ratio is not None
            else None
        ),
        "output_positive_rate": (
            float(output_positive_rate)
            if output_positive_rate is not None
            else None
        ),

        "input_summary": input_summary,

        "error_type": (
            type(error).__name__
            if error else None
        ),
        "error_message": (
            str(error)[:500]
            if error else None
        ),

        "created_by": None,
    }

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
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    df_input = None

    try:
        if csv_file is None:
            raise ValueError("Aucun fichier CSV fourni.")

        df_input = pd.read_csv(csv_file)

        # Si TARGET est présent, il n'est pas transmis au modèle
        df_input = df_input.drop(
            columns=["TARGET"],
            errors="ignore",
        )

        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in df_input.columns
        ]

        if missing_features:
            preview = ", ".join(missing_features[:10])

            raise ValueError(
                f"{len(missing_features)} features manquantes : "
                f"{preview}"
            )

        # Colonnes dans l'ordre appris pendant l'entraînement
        X_input = df_input[EXPECTED_FEATURES].copy()

        try:
            sample_path = save_production_sample(
                X_input,
                request_id,
            )

            logger.info(
                "request_id=%s monitoring_sample=%s",
                request_id,
                sample_path.name,
            )

        except Exception:
            logger.exception(
                "request_id=%s production_sample_save_failed",
                request_id,
            )

        probabilities = model.predict_proba(X_input)[:, 1]

        # Recommandé si tu as un seuil métier sauvegardé.
        predictions = (
            probabilities >= DECISION_THRESHOLD
        ).astype(int)

        result_df = df_input.copy()
        result_df["prediction"] = predictions
        result_df["probabilite_defaut"] = probabilities.round(4)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        # 1. Mesures techniques
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        n_rows = len(X_input)

        latency_per_row_ms = latency_ms / max(n_rows, 1)

        input_missing_ratio = float(
            X_input.isna().sum().sum() / X_input.size
        )

        output_positive_rate = float(
            (predictions == 1).mean()
        )


        # 2. Construire l'événement principal
        event = build_event(
            request_id=request_id,
            status="success",
            latency_ms=latency_ms,
            latency_per_row_ms=latency_per_row_ms,
            input_missing_ratio=input_missing_ratio,
            output_positive_rate=output_positive_rate,
            df_input=X_input,
            predictions=predictions,
            probabilities=probabilities,
        )


        # 3. Insérer la ligne parent D'ABORD
        log_prediction(event)


        # 4. Construire les statistiques par feature
        try:
            feature_stats = build_feature_statistics(
                X_input,
                request_id=request_id,
            )


            # 5. Insérer les lignes enfant ENSUITE
            log_feature_statistics(feature_stats)
        except Exception:
            logger.exception(
                "request_id=%s feature_monitoring_logging_failed",
                request_id,
    )

        try:
            log_prediction(event)
        except Exception:
            # La prédiction ne doit pas échouer seulement
            # parce que le logging est indisponible.
            logger.exception(
                "request_id=%s database_logging_failed",
                request_id,
            )

        logger.info(
            "request_id=%s status=success rows=%s latency_ms=%.2f",
            request_id,
            len(result_df),
            latency_ms,
        )

        status = (
            f"Prédiction terminée pour {len(result_df)} lignes. "
            f"request_id={request_id}. "
            f"Temps : {latency_ms:.2f} ms."
        )

        return result_df.head(100), status, str(OUTPUT_PATH)

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        n_rows = len(X_input)

        latency_per_row_ms = latency_ms / max(n_rows, 1)

        missing_ratio = float(
            X_input.isna().sum().sum() / X_input.size
        )

        positive_rate = float(
            (predictions == 1).mean()
        ) 

        event = build_event(
            request_id=request_id,
            status="error",
            latency_ms=latency_ms,
            df_input=df_input,
            error=exc,
            latency_per_row_ms=latency_per_row_ms,
            input_missing_ratio=missing_ratio,
            output_positive_rate=positive_rate,
        )

        try:
            log_prediction(event)
        except Exception:
            logger.exception(
                "request_id=%s error_logging_failed",
                request_id,
            )

        logger.exception(
            "request_id=%s status=error",
            request_id,
        )

        raise gr.Error(
            f"Erreur de prédiction. "
            f"Identifiant de requête : {request_id}."
        )

def save_production_sample(
    X_input: pd.DataFrame,
    request_id: str,
) -> Path:
    """
    Sauvegarde un échantillon limité de données d'entrée.
    À utiliser seulement si les données sont autorisées,
    pseudonymisées et sans colonnes directement identifiantes.
    """

    sample_size = min(
        MAX_PRODUCTION_SAMPLE_ROWS,
        len(X_input),
    )

    sample_df = X_input.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_path = (
        MONITORING_DIR
        / f"production_{timestamp}_{request_id}.parquet"
    )

    sample_df.to_parquet(
        output_path,
        index=False,
    )

    return output_path

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
    try:
        init_database()
        logger.info("Base de données de logs prête.")
    except Exception:
        # À discuter selon ton besoin :
        # fail-fast si audit obligatoire, sinon l'application continue.
        logger.exception(
            "Impossible d'initialiser la base de logs."
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )