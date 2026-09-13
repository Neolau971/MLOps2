from contextlib import contextmanager
import pytest
import modele.database as database


class FakeCursor:
    def __init__(self):
        self.executed = []
        self.executed_many = []

    def execute(self, query, params=None):
        self.executed.append(
            (query, params)
        )

    def executemany(self, query, records):
        self.executed_many.append(
            (query, records)
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

        return False

def test_get_connection_requires_database_url(
    monkeypatch,
):
    monkeypatch.setattr(
        database,
        "DATABASE_URL",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="DATABASE_URL est absente",
    ):
        with database.get_connection():
            pass

def test_init_database_executes_schema_queries(
    monkeypatch,
):
    fake_connection = FakeConnection()

    monkeypatch.setattr(
        database,
        "get_connection",
        lambda: fake_connection,
    )

    database.init_database()

    executed_queries = [
        query
        for query, _ in (
            fake_connection
            .cursor_instance
            .executed
        )
    ]

    joined_queries = "\n".join(
        executed_queries
    )

    assert "CREATE TABLE IF NOT EXISTS prediction_logs" in (
        joined_queries
    )

    assert (
        "CREATE TABLE IF NOT EXISTS "
        "feature_monitoring_stats"
    ) in joined_queries

    assert fake_connection.committed

def test_log_prediction_executes_insert(
    monkeypatch,
):
    fake_connection = FakeConnection()

    monkeypatch.setattr(
        database,
        "get_connection",
        lambda: fake_connection,
    )

    event = {
        "request_id": "test-request",
        "event_type": "prediction",
        "status": "success",
        "model_name": "random_forest_smote",
        "model_version": "1.0.0",
        "model_sha": None,
        "input_rows": 1,
        "input_columns": 2,
        "input_schema_hash": "hash",
        "input_file_hash": None,
        "prediction_positive_count": 0,
        "prediction_negative_count": 1,
        "probability_mean": 0.12,
        "probability_min": 0.12,
        "probability_max": 0.12,
        "decision_threshold": 0.5,
        "latency_ms": 10.0,
        "latency_per_row_ms": 10.0,
        "input_missing_ratio": 0.0,
        "output_positive_rate": 0.0,
        "input_summary": {"numeric_columns": 2},
        "error_type": None,
        "error_message": None,
        "created_by": None,
    }

    database.log_prediction(event)

    assert len(
        fake_connection.cursor_instance.executed
    ) == 1

    query, params = (
        fake_connection.cursor_instance.executed[0]
    )

    assert "INSERT INTO prediction_logs" in query
    assert params["request_id"] == "test-request"
    assert fake_connection.committed