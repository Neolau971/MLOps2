import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "random_forest_smote.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring_features.json"
)

IMPORTANCE_CSV_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "feature_importance.csv"
)

TOP_N_MODEL_FEATURES = 25


BUSINESS_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]


def get_feature_names(loaded_model) -> list[str]:
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    if hasattr(loaded_model, "named_steps"):
        estimator = loaded_model.named_steps.get(
            "model"
        )

        if (
            estimator is not None
            and hasattr(estimator, "feature_names_in_")
        ):
            return list(
                estimator.feature_names_in_
            )

    raise AttributeError(
        "Impossible de récupérer les noms des "
        "features du modèle."
    )


def get_random_forest(loaded_model):
    if hasattr(loaded_model, "feature_importances_"):
        return loaded_model

    if hasattr(loaded_model, "named_steps"):
        estimator = loaded_model.named_steps.get(
            "model"
        )

        if (
            estimator is not None
            and hasattr(
                estimator,
                "feature_importances_",
            )
        ):
            return estimator

    raise AttributeError(
        "Le modèle ne fournit pas "
        "feature_importances_."
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    feature_names = get_feature_names(model)
    random_forest = get_random_forest(model)

    importances = random_forest.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Incohérence entre noms et importances : "
            f"{len(feature_names)} features, "
            f"{len(importances)} importances."
        )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gini": importances,
            }
        )
        .sort_values(
            "importance_gini",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    IMPORTANCE_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_importance.to_csv(
        IMPORTANCE_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    top_model_features = (
        feature_importance
        .head(TOP_N_MODEL_FEATURES)["feature"]
        .tolist()
    )

    feature_set = set(feature_names)

    available_business_features = [
        feature
        for feature in BUSINESS_FEATURES
        if feature in feature_set
    ]

    unavailable_business_features = [
        feature
        for feature in BUSINESS_FEATURES
        if feature not in feature_set
    ]

    monitoring_features = list(
        dict.fromkeys(
            top_model_features
            + available_business_features
        )
    )

    payload = {
        "model_name": "random_forest_smote",
        "model_version": "1.0.0",
        "selection_method": (
            "top_25_random_forest_gini_importance"
            "_plus_available_business_features"
        ),
        "top_n_model_features": TOP_N_MODEL_FEATURES,
        "number_of_selected_features": len(
            monitoring_features
        ),
        "features": monitoring_features,
        "top_model_features": top_model_features,
        "business_features_available": (
            available_business_features
        ),
        "business_features_unavailable": (
            unavailable_business_features
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Importances sauvegardées : "
        f"{IMPORTANCE_CSV_PATH}"
    )

    print(
        f"Features de monitoring retenues : "
        f"{len(monitoring_features)}"
    )

    print("\n".join(monitoring_features))

    if unavailable_business_features:
        print(
            "\nVariables métier absentes du modèle "
            "après preprocessing :"
        )

        print(
            "\n".join(
                unavailable_business_features
            )
        )

    print(
        f"\nConfiguration créée : {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()