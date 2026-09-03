from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Rend importable le dossier "modele"
sys.path.insert(0, str(PROJECT_ROOT / "modele"))

from main import predict_from_csv, EXPECTED_FEATURES


def test_expected_features_is_not_empty():
    """Le modèle doit exposer au moins une feature attendue."""
    assert isinstance(EXPECTED_FEATURES, list)
    assert len(EXPECTED_FEATURES) > 0


def test_predict_from_csv_rejects_missing_columns(tmp_path):
    """Un CSV incomplet doit lever une erreur Gradio explicite."""

    incomplete_df = pd.DataFrame({
        EXPECTED_FEATURES[0]: [0],
    })

    csv_path = tmp_path / "incomplete_input.csv"
    incomplete_df.to_csv(csv_path, index=False)

    with pytest.raises(Exception):
        predict_from_csv(str(csv_path))