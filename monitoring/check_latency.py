import os
import json

import psycopg


DATABASE_URL = os.environ["DATABASE_URL"]

WARNING_THRESHOLD = 5.0
CRITICAL_THRESHOLD = 10.0


def get_status(value: float) -> str:
    if value >= CRITICAL_THRESHOLD:
        return "critical"

    if value >= WARNING_THRESHOLD:
        return "warning"

    return "ok"


def check_latency():
    query = """
    SELECT
        COUNT(*) AS n_requests,
        percentile_cont(0.95)
        WITHIN GROUP (
            ORDER BY latency_per_row_ms
        ) AS p95_latency_per_row_ms
    FROM prediction_logs
    WHERE status = 'success'
      AND created_at >= NOW() - INTERVAL '24 hours'
      AND latency_per_row_ms IS NOT NULL;
    """

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            n_requests, p95_latency = cursor.fetchone()

            if not n_requests or p95_latency is None:
                print("Pas assez de données pour vérifier la latence.")
                return

            status = get_status(float(p95_latency))

            cursor.execute(
                """
                INSERT INTO monitoring_results (
                    check_type,
                    status,
                    model_name,
                    observed_value,
                    warning_threshold,
                    critical_threshold,
                    details,
                    message
                )
                VALUES (
                    'latency_p95_per_row',
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s
                );
                """,
                (
                    status,
                    "random_forest_smote",
                    float(p95_latency),
                    WARNING_THRESHOLD,
                    CRITICAL_THRESHOLD,
                    json.dumps({
                        "window": "24 hours",
                        "n_requests": n_requests,
                    }),
                    (
                        f"p95 latency/row = "
                        f"{float(p95_latency):.3f} ms"
                    ),
                ),
            )

            connection.commit()

            print(
                f"Statut={status} | "
                f"p95={float(p95_latency):.3f} ms/ligne"
            )


if __name__ == "__main__":
    check_latency()