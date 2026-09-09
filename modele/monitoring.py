import json
import numpy as np
import pandas as pd


def build_feature_statistics(
    df: pd.DataFrame,
    request_id: str,
    max_categories: int = 20,
    n_bins: int = 10,
) -> list[dict]:
    records = []

    for feature_name in df.columns:
        series = df[feature_name]
        n_values = int(len(series))
        n_missing = int(series.isna().sum())

        record = {
            "request_id": request_id,
            "feature_name": feature_name,
            "n_values": n_values,
            "n_missing": n_missing,
            "mean_value": None,
            "std_value": None,
            "min_value": None,
            "max_value": None,
            "category_counts": None,
            "histogram_counts": None,
            "histogram_edges": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            record["feature_type"] = "numeric"

            valid_values = series.dropna().astype(float)

            if not valid_values.empty:
                counts, edges = np.histogram(
                    valid_values,
                    bins=n_bins,
                )

                record["mean_value"] = float(valid_values.mean())
                record["std_value"] = float(valid_values.std(ddof=0))
                record["min_value"] = float(valid_values.min())
                record["max_value"] = float(valid_values.max())
                record["histogram_counts"] = json.dumps(
                    counts.tolist()
                )
                record["histogram_edges"] = json.dumps(
                    edges.tolist()
                )

        else:
            record["feature_type"] = "categorical"

            counts = (
                series.fillna("__MISSING__")
                .astype(str)
                .value_counts()
                .head(max_categories)
                .to_dict()
            )

            record["category_counts"] = json.dumps(counts)

        records.append(record)

    return records