import numpy as np
import pandas as pd

from modele.main import build_event


def test_build_event_success_contains_metrics():
    df_input = pd.DataFrame(
        {
            "feature_a": [1.0, None],
            "feature_b": [0.0, 1.0],
        }
    )

    predictions = np.array([0, 1])
    probabilities = np.array([0.12, 0.85])

    event = build_event(
        request_id="test-request-id",
        status="success",
        latency_ms=123.4,
        latency_per_row_ms=61.7,
        input_missing_ratio=0.25,
        output_positive_rate=0.5,
        df_input=df_input,
        predictions=predictions,
        probabilities=probabilities,
    )

    assert event["request_id"] == "test-request-id"
    assert event["status"] == "success"

    assert event["input_rows"] == 2
    assert event["input_columns"] == 2
    assert event["input_schema_hash"] is not None

    assert event["prediction_positive_count"] == 1
    assert event["prediction_negative_count"] == 1

    assert event["probability_mean"] == 0.485
    assert event["probability_min"] == 0.12
    assert event["probability_max"] == 0.85

    assert event["latency_ms"] == 123.4
    assert event["latency_per_row_ms"] == 61.7
    assert event["input_missing_ratio"] == 0.25
    assert event["output_positive_rate"] == 0.5

    assert event["error_type"] is None
    assert event["error_message"] is None


def test_build_event_error_without_input():
    error = ValueError("CSV invalide")

    event = build_event(
        request_id="error-request",
        status="error",
        latency_ms=5.0,
        error=error,
    )

    assert event["status"] == "error"
    assert event["input_rows"] is None
    assert event["input_columns"] is None
    assert event["input_summary"] is None

    assert event["prediction_positive_count"] is None
    assert event["prediction_negative_count"] is None

    assert event["probability_mean"] is None
    assert event["probability_min"] is None
    assert event["probability_max"] is None

    assert event["error_type"] == "ValueError"
    assert "CSV invalide" in event["error_message"]


def test_build_event_truncates_long_error_message():
    long_error = RuntimeError("x" * 800)

    event = build_event(
        request_id="long-error",
        status="error",
        latency_ms=1.0,
        error=long_error,
    )

    assert len(event["error_message"]) == 500