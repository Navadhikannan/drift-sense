from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance


# ============================================================
# DRIFT-SENSE AI PHASE
# AI-2: TRAIN CANDIDATE RERANKER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    BASE_DIR
    / "results"
    / "ai"
    / "candidate_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "ai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "ai_reranker.joblib"
)

IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

TEST_RESULTS_PATH = (
    OUTPUT_DIR
    / "test_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "rank",
    "candidate_x",
    "candidate_y",
    "template_score",
    "gray_score",
    "edge_score",
    "gradient_score",
    "structural_score",
    "center_distance",
]


# ============================================================
# SAMPLE SPLIT
# ============================================================

TRAIN_SAMPLES = {
    f"sample_{i:03d}"
    for i in range(1, 21)
}

TEST_SAMPLES = {
    f"sample_{i:03d}"
    for i in range(21, 31)
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "DRIFT-SENSE AI PHASE 6.2"
    )
    print(
        "RANDOM FOREST CANDIDATE RERANKER"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print()
    print(
        f"Total candidate rows : "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            "sample",
            "noise_level",
            "label",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Split by sample identity
    # --------------------------------------------------------

    train_df = df[
        df["sample"].isin(
            TRAIN_SAMPLES
        )
    ].copy()

    test_df = df[
        df["sample"].isin(
            TEST_SAMPLES
        )
    ].copy()

    print()
    print(
        "TRAIN / TEST SPLIT"
    )
    print("-" * 75)

    print(
        f"Training samples : "
        f"{len(TRAIN_SAMPLES)}"
    )

    print(
        f"Testing samples  : "
        f"{len(TEST_SAMPLES)}"
    )

    print(
        f"Training rows    : "
        f"{len(train_df)}"
    )

    print(
        f"Testing rows     : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print(
        "TRAIN LABEL DISTRIBUTION"
    )
    print("-" * 75)

    print(
        train_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "TEST LABEL DISTRIBUTION"
    )
    print("-" * 75)

    print(
        test_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "label"
    ]

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print()
    print(
        "TRAINING RANDOM FOREST..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "[OK] Model training complete."
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "CLASSIFICATION RESULTS"
    )
    print("=" * 75)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if len(
        np.unique(y_test)
    ) == 2:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        print()
        print(
            f"ROC-AUC : {auc:.4f}"
        )

    else:

        auc = float("nan")

        print()
        print(
            "ROC-AUC unavailable."
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    test_output = test_df[
        [
            "sample",
            "noise_level",
            "rank",
            "candidate_x",
            "candidate_y",
            "gt_x",
            "gt_y",
            "distance_to_gt",
            "label",
        ]
    ].copy()

    test_output[
        "predicted_label"
    ] = predictions

    test_output[
        "probability"
    ] = probabilities

    test_output.to_csv(
        TEST_RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance":
                model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False
    )

    print()
    print(
        "=" * 75
    )

    print(
        "FEATURE IMPORTANCE"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH
    )

    print()
    print(
        "=" * 75
    )

    print(
        "AI RERANKER TRAINING COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Predictions : {TEST_RESULTS_PATH}"
    )

    print(
        f"Importance  : {IMPORTANCE_PATH}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()