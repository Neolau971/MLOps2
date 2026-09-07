import json
import os
from contextlib import contextmanager

import psycopg


DATABASE_URL = os.getenv("DATABASE_URL")


@contextmanager
def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "La variable d'environnement DATABASE_URL est absente."
        )

    connection = psycopg.connect(DATABASE_URL)

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database():
    query = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id BIGSERIAL PRIMARY KEY,

        request_id UUID NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        event_type VARCHAR(50) NOT NULL DEFAULT 'prediction',
        status VARCHAR(20) NOT NULL,

        model_name VARCHAR(150) NOT NULL,
        model_version VARCHAR(100),
        model_sha VARCHAR(100),

        input_rows INTEGER,
        input_columns INTEGER,
        input_schema_hash VARCHAR(64),
        input_file_hash VARCHAR(64),

        prediction_positive_count INTEGER,
        prediction_negative_count INTEGER,
        probability_mean DOUBLE PRECISION,
        probability_min DOUBLE PRECISION,
        probability_max DOUBLE PRECISION,
        decision_threshold DOUBLE PRECISION,

        latency_ms DOUBLE PRECISION NOT NULL,

        input_summary JSONB,
        error_type VARCHAR(150),
        error_message TEXT,

        created_by VARCHAR(150)
    );
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)


def log_prediction(event: dict):
    query = """
    INSERT INTO prediction_logs (
        request_id,
        event_type,
        status,

        model_name,
        model_version,
        model_sha,

        input_rows,
        input_columns,
        input_schema_hash,
        input_file_hash,

        prediction_positive_count,
        prediction_negative_count,
        probability_mean,
        probability_min,
        probability_max,
        decision_threshold,

        latency_ms,
        input_summary,

        error_type,
        error_message,
        created_by
    )
    VALUES (
        %(request_id)s,
        %(event_type)s,
        %(status)s,

        %(model_name)s,
        %(model_version)s,
        %(model_sha)s,

        %(input_rows)s,
        %(input_columns)s,
        %(input_schema_hash)s,
        %(input_file_hash)s,

        %(prediction_positive_count)s,
        %(prediction_negative_count)s,
        %(probability_mean)s,
        %(probability_min)s,
        %(probability_max)s,
        %(decision_threshold)s,

        %(latency_ms)s,
        %(input_summary)s::jsonb,

        %(error_type)s,
        %(error_message)s,
        %(created_by)s
    );
    """

    event_to_insert = {
        **event,
        "input_summary": json.dumps(event.get("input_summary", {})),
    }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, event_to_insert)