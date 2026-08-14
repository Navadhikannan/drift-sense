import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# DRIFT-SENSE AI-V2
# RELATIVE CANDIDATE RANKING RERANKER
# ============================================================

INPUT_FILE = "results/ai_v2/candidate_ranking_dataset.csv"
OUTPUT_DIR = "results/ai_v2"

MODEL_FILE = os.path.join(OUTPUT_DIR, "ai_v2_reranker.joblib")
PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "ai_v2_test_predictions.csv"
)
IMPORTANCE_FILE = os.path.join(
    OUTPUT_DIR,
    "ai_v2_feature_importance.csv"
)


FEATURES = [
    "template_gap",
    "gray_gap",
    "edge_gap",
    "gradient_gap",
    "structural_gap",
    "combined_score",
    "combined_gap",
    "normalized_rank",
    "distance_from_template_best",
    "nearest_candidate_distance",
    "neighborhood_density_50",
    "neighborhood_density_100",
]


def main():

    print("=" * 79)
    print("DRIFT-SENSE AI-V2")
    print("RELATIVE CANDIDATE RANKING RERANKER")
    print("=" * 79)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print()
    print("Total candidate rows :", len(df))
    print("Candidate groups     :", df.groupby(
        ["sample", "noise_level"]
    ).ngroups)

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required features:\n" +
            "\n".join(missing)
        )

    # --------------------------------------------------------
    # GROUP SPLIT
    #
    # IMPORTANT:
    # All candidates belonging to one sample/noise pair
    # stay together.
    # --------------------------------------------------------

    groups = (
        df["sample"].astype(str)
        + "_"
        + df["noise_level"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["label"],
            groups=groups
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print()
    print("TRAIN / TEST SPLIT")
    print("-" * 79)

    print("Training groups :", train_df.groupby(
        ["sample", "noise_level"]
    ).ngroups)

    print("Testing groups  :", test_df.groupby(
        ["sample", "noise_level"]
    ).ngroups)

    print("Training rows   :", len(train_df))
    print("Testing rows    :", len(test_df))

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("TRAIN LABEL DISTRIBUTION")
    print("-" * 79)
    print(train_df["label"].value_counts().sort_index())

    print()
    print("TEST LABEL DISTRIBUTION")
    print("-" * 79)
    print(test_df["label"].value_counts().sort_index())

    # --------------------------------------------------------
    # FEATURES / LABELS
    # --------------------------------------------------------

    X_train = train_df[FEATURES]
    y_train = train_df["label"]

    X_test = test_df[FEATURES]
    y_test = test_df["label"]

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print()
    print("TRAINING AI-V2 RANDOM FOREST...")
    print("-" * 79)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[OK] AI-V2 model training complete.")

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print()
    print("=" * 79)
    print("AI-V2 CLASSIFICATION RESULTS")
    print("=" * 79)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("CONFUSION MATRIX")
    print("-" * 79)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if len(y_test.unique()) == 2:
        auc = roc_auc_score(
            y_test,
            probabilities
        )

        print()
        print("ROC-AUC :", f"{auc:.4f}")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("=" * 79)
    print("AI-V2 FEATURE IMPORTANCE")
    print("=" * 79)

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "version": "AI-V2"
        },
        MODEL_FILE
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = test_df.copy()

    prediction_df["prediction"] = predictions
    prediction_df["probability"] = probabilities

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 79)
    print("AI-V2 TRAINING COMPLETE")
    print("=" * 79)

    print("Model       :", os.path.abspath(MODEL_FILE))
    print("Predictions :", os.path.abspath(PREDICTIONS_FILE))
    print("Importance  :", os.path.abspath(IMPORTANCE_FILE))

    print("=" * 79)


if __name__ == "__main__":
    main()