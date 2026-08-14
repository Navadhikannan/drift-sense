from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# DRIFT-SENSE AI PHASE 6.3
# AI CANDIDATE RERANKING EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "results"
    / "ai"
    / "ai_reranker.joblib"
)

DATASET_PATH = (
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "localization_results.csv"
)


# ============================================================
# TEST SAMPLES
# ============================================================

TEST_SAMPLES = {
    f"sample_{i:03d}"
    for i in range(21, 31)
}


# ============================================================
# FEATURES
# ============================================================

def main():

    print("=" * 80)
    print(
        "DRIFT-SENSE AI PHASE 6.3"
    )
    print(
        "AI CANDIDATE RERANKING"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    package = joblib.load(
        MODEL_PATH
    )

    model = package["model"]

    features = package["features"]

    # --------------------------------------------------------
    # Load candidate dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    test_df = df[
        df["sample"].isin(
            TEST_SAMPLES
        )
    ].copy()

    print()
    print(
        f"Test samples : "
        f"{len(TEST_SAMPLES)}"
    )

    print(
        f"Candidate rows : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Predict probability for every candidate
    # --------------------------------------------------------

    test_df[
        "ai_probability"
    ] = model.predict_proba(
        test_df[features]
    )[:, 1]

    # --------------------------------------------------------
    # Select highest probability candidate
    # for each sample/noise condition
    # --------------------------------------------------------

    results = []

    groups = test_df.groupby(
        [
            "sample",
            "noise_level"
        ]
    )

    for (
        sample,
        noise_level
    ), group in groups:

        group = group.sort_values(
            "ai_probability",
            ascending=False
        )

        winner = group.iloc[0]

        predicted_x = float(
            winner["candidate_x"]
        )

        predicted_y = float(
            winner["candidate_y"]
        )

        gt_x = float(
            winner["gt_x"]
        )

        gt_y = float(
            winner["gt_y"]
        )

        error = np.sqrt(
            (predicted_x - gt_x) ** 2
            +
            (predicted_y - gt_y) ** 2
        )

        # Original template matching winner
        original = group.sort_values(
            "template_score",
            ascending=False
        ).iloc[0]

        original_error = np.sqrt(
            (
                float(
                    original["candidate_x"]
                )
                -
                gt_x
            ) ** 2
            +
            (
                float(
                    original["candidate_y"]
                )
                -
                gt_y
            ) ** 2
        )

        results.append(
            {
                "sample":
                    sample,

                "noise_level":
                    noise_level,

                "gt_x":
                    gt_x,

                "gt_y":
                    gt_y,

                "ai_x":
                    predicted_x,

                "ai_y":
                    predicted_y,

                "ai_probability":
                    float(
                        winner[
                            "ai_probability"
                        ]
                    ),

                "ai_error":
                    error,

                "baseline_x":
                    float(
                        original[
                            "candidate_x"
                        ]
                    ),

                "baseline_y":
                    float(
                        original[
                            "candidate_y"
                        ]
                    ),

                "baseline_error":
                    original_error,

                "improvement":
                    original_error
                    - error
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        [
            "noise_level",
            "sample"
        ]
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "AI LOCALIZATION RESULTS"
    )
    print("=" * 80)

    print()

    print(
        f"Evaluated cases : "
        f"{len(results_df)}"
    )

    print()

    print(
        f"AI mean error : "
        f"{results_df['ai_error'].mean():.3f} px"
    )

    print(
        f"AI median error : "
        f"{results_df['ai_error'].median():.3f} px"
    )

    print(
        f"AI worst error : "
        f"{results_df['ai_error'].max():.3f} px"
    )

    print()

    print(
        f"Baseline mean error : "
        f"{results_df['baseline_error'].mean():.3f} px"
    )

    print(
        f"Baseline median error : "
        f"{results_df['baseline_error'].median():.3f} px"
    )

    print(
        f"Baseline worst error : "
        f"{results_df['baseline_error'].max():.3f} px"
    )

    # ========================================================
    # PASS RATES
    # ========================================================

    print()
    print(
        "PASS RATES"
    )
    print("-" * 80)

    for threshold in [
        1,
        2,
        5,
        20,
        50
    ]:

        ai_pass = (
            results_df["ai_error"]
            <= threshold
        ).sum()

        baseline_pass = (
            results_df["baseline_error"]
            <= threshold
        ).sum()

        total = len(
            results_df
        )

        print(
            f"@ {threshold:>2}px : "
            f"AI {ai_pass}/{total} "
            f"({100 * ai_pass / total:.2f}%)   |   "
            f"Baseline {baseline_pass}/{total} "
            f"({100 * baseline_pass / total:.2f}%)"
        )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    print()
    print(
        "AI IMPROVEMENT"
    )
    print("-" * 80)

    improved = (
        results_df["improvement"]
        > 0
    ).sum()

    worse = (
        results_df["improvement"]
        < 0
    ).sum()

    equal = (
        results_df["improvement"]
        == 0
    ).sum()

    print(
        f"Improved : "
        f"{improved}/{len(results_df)}"
    )

    print(
        f"Worse    : "
        f"{worse}/{len(results_df)}"
    )

    print(
        f"Equal    : "
        f"{equal}/{len(results_df)}"
    )

    # ========================================================
    # WORST CASES
    # ========================================================

    print()
    print(
        "WORST AI CASES"
    )
    print("-" * 80)

    print(
        results_df[
            [
                "sample",
                "noise_level",
                "ai_error",
                "baseline_error",
                "ai_probability"
            ]
        ]
        .sort_values(
            "ai_error",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # NOISE SUMMARY
    # ========================================================

    print()
    print(
        "RESULTS BY NOISE LEVEL"
    )
    print("-" * 80)

    summary = (
        results_df
        .groupby("noise_level")
        .agg(
            samples=(
                "sample",
                "count"
            ),

            ai_mean_error=(
                "ai_error",
                "mean"
            ),

            ai_median_error=(
                "ai_error",
                "median"
            ),

            ai_worst_error=(
                "ai_error",
                "max"
            ),

            baseline_mean_error=(
                "baseline_error",
                "mean"
            ),

            baseline_median_error=(
                "baseline_error",
                "median"
            ),

            baseline_worst_error=(
                "baseline_error",
                "max"
            )
        )
    )

    print(
        summary.to_string()
    )

    print()
    print("=" * 80)
    print(
        "AI LOCALIZATION EVALUATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()