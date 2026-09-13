import json

import numpy as np
import pandas as pd

from modele.monitoring import (
    build_feature_statistics,
)


def test_build_feature_statistics_returns_one_record_per_column():
    df_input = pd.DataFrame(
        {
            "income": [1000.0, 2000.0, np.nan],
            "is_working": [1, 0, 1],
            "category": ["A", "B", None],
        }
    )

    records = build_feature_statistics(
        df_input,
        request_id="test-request-id",
    )

    assert len(records) == 3

    feature_names = {
        record["feature_name"]
        for record in records
    }

    assert feature_names == {
        "income",
        "is_working",
        "category",
    }

    for record in records:
        assert (
            record["request_id"]
            == "test-request-id"
        )
        assert record["n_values"] == 3


def test_build_feature_statistics_numeric_values():
    df_input = pd.DataFrame(
        {
            "amount": [10.0, 20.0, np.nan, 30.0],
        }
    )

    records = build_feature_statistics(
        df_input,
        request_id="numeric-request",
    )

    assert len(records) == 1

    stats = records[0]

    assert stats["request_id"] == "numeric-request"
    assert stats["feature_name"] == "amount"
    assert stats["feature_type"] == "numeric"

    assert stats["n_values"] == 4
    assert stats["n_missing"] == 1

    assert stats["mean_value"] == 20.0
    assert stats["min_value"] == 10.0
    assert stats["max_value"] == 30.0

    assert stats["std_value"] is not None

    assert stats["category_counts"] is None

    assert stats["histogram_counts"] is not None
    assert stats["histogram_edges"] is not None

    histogram_counts = json.loads(
        stats["histogram_counts"]
    )

    histogram_edges = json.loads(
        stats["histogram_edges"]
    )

    assert sum(histogram_counts) == 3

    # 10 bins → 11 bornes.
    assert len(histogram_counts) == 10
    assert len(histogram_edges) == 11

    assert histogram_edges[0] == 10.0
    assert histogram_edges[-1] == 30.0

def test_build_feature_statistics_categorical_values():
    df_input = pd.DataFrame(
        {
            "income_type": [
                "Working",
                "Pensioner",
                "Working",
                None,
            ],
        }
    )

    records = build_feature_statistics(
        df_input,
        request_id="categorical-request",
    )

    assert len(records) == 1

    stats = records[0]

    assert stats["request_id"] == "categorical-request"
    assert stats["feature_name"] == "income_type"
    assert stats["feature_type"] == "categorical"

    assert stats["n_values"] == 4
    assert stats["n_missing"] == 1

    assert stats["mean_value"] is None
    assert stats["std_value"] is None
    assert stats["min_value"] is None
    assert stats["max_value"] is None

    assert stats["histogram_counts"] is None
    assert stats["histogram_edges"] is None

    category_counts = json.loads(
        stats["category_counts"]
    )

    assert category_counts == {
        "Working": 2,
        "Pensioner": 1,
        "__MISSING__": 1,
    }