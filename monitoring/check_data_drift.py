from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import os
import psycopg

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MONITORING_DIR = PROJECT_ROOT / "monitoring_data"
REPORTS_DIR = PROJECT_ROOT / "monitoring_reports"
WARNING_DRIFT_SHARE = 0.10
CRITICAL_DRIFT_SHARE = 0.25

REFERENCE_PATH = (
    ARTIFACTS_DIR
    / "reference_data.parquet"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_URL = os.getenv("DATABASE_URL")

MODEL_NAME = "random_forest_smote"
MODEL_VERSION = "1.0.0"


def load_reference() -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Référence introuvable : {REFERENCE_PATH}"
        )

    return pd.read_parquet(REFERENCE_PATH)


def load_current_data() -> pd.DataFrame:
    production_files = sorted(
        MONITORING_DIR.glob("production_*.parquet")
    )

    if not production_files:
        raise FileNotFoundError(
            "Aucun fichier de production à analyser."
        )

    frames = [
        pd.read_parquet(path)
        for path in production_files
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_features = list(reference_df.columns)

    missing_columns = [
        column
        for column in expected_features
        if column not in current_df.columns
    ]

    if missing_columns:
        preview = ", ".join(missing_columns[:15])

        raise ValueError(
            "Colonnes absentes des données courantes : "
            f"{preview}"
        )

    # Ignore les colonnes potentiellement supplémentaires
    # et réordonne selon la référence.
    current_df = current_df[expected_features].copy()
    reference_df = reference_df[expected_features].copy()

    return reference_df, current_df


def build_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_path: Path,
):
    report = Report([
        DataDriftPreset(),
    ])

    result = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    result.save_html(report_path)

    return result

def get_nested_value(data: dict, *keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

    return current if current is not None else default

def get_drift_summary(result) -> dict:
    """
    Extrait un résumé de drift exploitable dans PostgreSQL/Grafana.
    Garde aussi la sortie brute pour faciliter l'adaptation
    selon la version exacte d'Evidently.
    """

    raw = result.dict()

    # Chemins possibles selon versions / format d'Evidently.
    metrics = raw.get("metrics", [])
    drift_metric = None

    for metric in metrics:
        metric_id = str(
            metric.get("metric_id", "")
        ).lower()

        metric_name = str(
            metric.get("metric", "")
        ).lower()

        if (
            "dataset_drift" in metric_id
            or "datadrift" in metric_id
            or "datasetdrift" in metric_name
        ):
            drift_metric = metric
            break

    # Valeurs par défaut, évitent une erreur si la structure diffère.
    dataset_drift = False
    number_of_drifted_columns = 0
    share_of_drifted_columns = 0.0

    if drift_metric is not None:
        value = drift_metric.get("value", drift_metric)

        if isinstance(value, dict):
            dataset_drift = bool(
                value.get(
                    "dataset_drift",
                    value.get("drift_detected", False),
                )
            )

            number_of_drifted_columns = int(
                value.get(
                    "number_of_drifted_columns",
                    value.get("drifted_columns_count", 0),
                )
                or 0
            )

            share_of_drifted_columns = float(
                value.get(
                    "share_of_drifted_columns",
                    value.get("drifted_columns_share", 0.0),
                )
                or 0.0
            )

    return {
        "dataset_drift": dataset_drift,
        "number_of_drifted_columns": number_of_drifted_columns,
        "share_of_drifted_columns": share_of_drifted_columns,
        "raw": raw,
    }

def get_drift_status(share: float) -> str:
    if share >= CRITICAL_DRIFT_SHARE:
        return "critical"

    if share >= WARNING_DRIFT_SHARE:
        return "warning"

    return "ok"

def save_drift_result(
    summary: dict,
    reference_rows: int,
    current_rows: int,
    report_path: Path,
):
    """
    Enregistre le résumé Evidently dans PostgreSQL afin que
    Grafana puisse l'afficher dans le temps.
    """

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL est absente."
        )

    share = float(
        summary["share_of_drifted_columns"]
    )

    status = get_drift_status(share)

    details = {
        "dataset_drift": summary["dataset_drift"],
        "number_of_drifted_columns": (
            summary["number_of_drifted_columns"]
        ),
        "share_of_drifted_columns": share,
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "report_file": report_path.name,
    }

    message = (
        f"{details['number_of_drifted_columns']} feature(s) "
        f"en drift sur la fenêtre analysée ; "
        f"part = {share:.2%}."
    )

    query = """
    INSERT INTO monitoring_results (
        check_type,
        status,
        model_name,
        model_version,
        observed_value,
        warning_threshold,
        critical_threshold,
        details,
        message
    )
    VALUES (
        'data_drift',
        %(status)s,
        %(model_name)s,
        %(model_version)s,
        %(observed_value)s,
        %(warning_threshold)s,
        %(critical_threshold)s,
        %(details)s::jsonb,
        %(message)s
    );
    """

    params = {
        "status": status,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "observed_value": share,
        "warning_threshold": WARNING_DRIFT_SHARE,
        "critical_threshold": CRITICAL_DRIFT_SHARE,
        "details": json.dumps(details),
        "message": message,
    }

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

        connection.commit()

    print(
        "Résultat drift enregistré dans PostgreSQL : "
        f"status={status}, share={share:.2%}"
    )

def main():
    reference_df = load_reference()
    current_df = load_current_data()

    reference_df, current_df = validate_columns(
        reference_df,
        current_df,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    report_path = (
        REPORTS_DIR
        / f"data_drift_report_{timestamp}.html"
    )

    json_path = (
        REPORTS_DIR
        / f"data_drift_report_{timestamp}.json"
    )

    result = build_report(
        reference_df,
        current_df,
        report_path,
    )

    summary = get_drift_summary(result)

    with open(
    json_path,
    "w",
    encoding="utf-8",
    ) as file:
        json.dump(
            summary["raw"],
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    save_drift_result(
        summary=summary,
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        report_path=report_path,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        f"Rapport HTML créé : {report_path}"
    )
    print(
        f"Résultat JSON créé : {json_path}"
    )
    print(
        f"Reference rows={len(reference_df)}, "
        f"current rows={len(current_df)}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"Erreur lors du contrôle de drift : {exc}",
            file=sys.stderr,
        )
        raise