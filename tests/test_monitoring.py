from pathlib import Path

import pandas as pd

from modele.main import save_production_sample


def test_save_production_sample_creates_parquet(
    monkeypatch,
    tmp_path: Path,
):
    import modele.main as app

    monkeypatch.setattr(
        app,
        "MONITORING_DIR",
        tmp_path,
    )

    input_df = pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": range(20, 40),
        }
    )

    output_path = save_production_sample(
        input_df,
        request_id="request-test",
    )

    assert output_path.exists()
    assert output_path.suffix == ".parquet"
    assert "request-test" in output_path.name

    saved_df = pd.read_parquet(output_path)

    assert len(saved_df) == len(input_df)
    assert list(saved_df.columns) == [
        "feature_a",
        "feature_b",
    ]


def test_save_production_sample_limits_rows(
    monkeypatch,
    tmp_path: Path,
):
    import modele.main as app

    monkeypatch.setattr(
        app,
        "MONITORING_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        app,
        "MAX_PRODUCTION_SAMPLE_ROWS",
        3,
    )

    input_df = pd.DataFrame(
        {
            "feature_a": range(10),
        }
    )

    output_path = save_production_sample(
        input_df,
        request_id="limited-request",
    )

    saved_df = pd.read_parquet(output_path)

    assert len(saved_df) == 3