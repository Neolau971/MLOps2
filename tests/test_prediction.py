from pathlib import Path

import pandas as pd
import pytest

from modele.main import (
    EXPECTED_FEATURES,
    predict_dataframe,
    predict_from_csv,
)


def make_valid_input(
    rows: int = 2,
) -> pd.DataFrame:
    """
    Construit un DataFrame minimal compatible avec les
    features attendues par le modèle.
    """

    return pd.DataFrame(
        {
            feature: [0] * rows
            for feature in EXPECTED_FEATURES
        }
    )


def test_expected_features_is_not_empty():
    assert isinstance(EXPECTED_FEATURES, list)
    assert len(EXPECTED_FEATURES) > 0


def test_predict_dataframe_returns_expected_outputs():
    input_df = make_valid_input(rows=3)

    (
        result_df,
        X_input,
        predictions,
        probabilities,
    ) = predict_dataframe(input_df)

    assert len(result_df) == 3
    assert X_input.shape == (3, len(EXPECTED_FEATURES))

    assert list(X_input.columns) == EXPECTED_FEATURES

    assert "prediction" in result_df.columns
    assert "probabilite_defaut" in result_df.columns

    assert len(predictions) == 3
    assert len(probabilities) == 3

    assert set(predictions).issubset({0, 1})

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()


def test_predict_dataframe_accepts_target_column():
    input_df = make_valid_input(rows=2)
    input_df["TARGET"] = [0, 1]

    result_df, X_input, _, _ = predict_dataframe(
        input_df
    )

    assert "TARGET" not in result_df.columns
    assert "TARGET" not in X_input.columns


def test_predict_dataframe_rejects_empty_dataframe():
    with pytest.raises(
        ValueError,
        match="aucune ligne",
    ):
        predict_dataframe(pd.DataFrame())


def test_predict_dataframe_rejects_missing_features():
    incomplete_df = pd.DataFrame(
        {
            EXPECTED_FEATURES[0]: [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="features manquantes",
    ):
        predict_dataframe(incomplete_df)


def test_predict_from_csv_rejects_no_file():
    with pytest.raises(Exception):
        predict_from_csv(None)


def test_predict_from_csv_rejects_missing_columns(
    tmp_path: Path,
):
    incomplete_df = pd.DataFrame(
        {
            EXPECTED_FEATURES[0]: [0],
        }
    )

    csv_path = tmp_path / "incomplete_input.csv"

    incomplete_df.to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(Exception):
        predict_from_csv(str(csv_path))

def test_predict_from_csv_success(
    monkeypatch,
    tmp_path: Path,
):
    import modele.main as app

    input_df = make_valid_input(rows=2)

    csv_path = tmp_path / "valid_input.csv"
    output_path = tmp_path / "predictions.csv"

    input_df.to_csv(
        csv_path,
        index=False,
    )

    logged_events = []
    logged_stats = []

    monkeypatch.setattr(
        app,
        "OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        app,
        "OUTPUT_PATH",
        output_path,
    )

    monkeypatch.setattr(
        app,
        "save_production_sample",
        lambda X_input, request_id: (
            tmp_path / f"production_{request_id}.parquet"
        ),
    )

    monkeypatch.setattr(
        app,
        "log_prediction",
        lambda event: logged_events.append(event),
    )

    monkeypatch.setattr(
        app,
        "build_feature_statistics",
        lambda X_input, request_id: [
            {
                "request_id": request_id,
                "feature_name": "feature_test",
            }
        ],
    )

    monkeypatch.setattr(
        app,
        "log_feature_statistics",
        lambda stats: logged_stats.extend(stats),
    )

    preview_df, status, produced_path = (
        app.predict_from_csv(str(csv_path))
    )

    assert output_path.exists()
    assert produced_path == str(output_path)

    assert len(preview_df) == 2
    assert "prediction" in preview_df.columns
    assert "probabilite_defaut" in preview_df.columns

    assert "Prédiction terminée" in status
    assert len(logged_events) == 1
    assert logged_events[0]["status"] == "success"

    assert len(logged_stats) == 1

def test_prediction_succeeds_when_sample_save_fails(
    monkeypatch,
    tmp_path: Path,
):
    import modele.main as app

    input_df = make_valid_input(rows=2)

    csv_path = tmp_path / "valid_input.csv"
    output_path = tmp_path / "predictions.csv"

    input_df.to_csv(
        csv_path,
        index=False,
    )

    def raise_sample_error(*args, **kwargs):
        raise OSError("Disque indisponible")

    monkeypatch.setattr(
        app,
        "OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        app,
        "OUTPUT_PATH",
        output_path,
    )

    monkeypatch.setattr(
        app,
        "save_production_sample",
        raise_sample_error,
    )

    monkeypatch.setattr(
        app,
        "log_prediction",
        lambda event: None,
    )

    monkeypatch.setattr(
        app,
        "build_feature_statistics",
        lambda X_input, request_id: [],
    )

    monkeypatch.setattr(
        app,
        "log_feature_statistics",
        lambda stats: None,
    )

    preview_df, status, produced_path = (
        app.predict_from_csv(str(csv_path))
    )

    assert output_path.exists()
    assert len(preview_df) == 2
    assert "Prédiction terminée" in status
    assert produced_path == str(output_path)


def test_prediction_succeeds_when_feature_stats_fail(
    monkeypatch,
    tmp_path: Path,
):
    import modele.main as app

    input_df = make_valid_input(rows=2)

    csv_path = tmp_path / "valid_input.csv"
    output_path = tmp_path / "predictions.csv"

    input_df.to_csv(
        csv_path,
        index=False,
    )

    monkeypatch.setattr(
        app,
        "OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        app,
        "OUTPUT_PATH",
        output_path,
    )

    monkeypatch.setattr(
        app,
        "save_production_sample",
        lambda X_input, request_id: (
            tmp_path / "production.parquet"
        ),
    )

    monkeypatch.setattr(
        app,
        "log_prediction",
        lambda event: None,
    )

    def raise_stats_error(*args, **kwargs):
        raise RuntimeError(
            "Monitoring indisponible"
        )

    monkeypatch.setattr(
        app,
        "build_feature_statistics",
        raise_stats_error,
    )

    preview_df, status, produced_path = (
        app.predict_from_csv(str(csv_path))
    )

    assert output_path.exists()
    assert len(preview_df) == 2
    assert "Prédiction terminée" in status
    assert produced_path == str(output_path)