import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# DRIFT-SENSE AI-V2 PHASE 6.4
# AI RELATIVE CANDIDATE LOCALIZATION
# ============================================================

DATASET_FILE = (
    "results/ai_v2/candidate_ranking_dataset.csv"
)

MODEL_FILE = (
    "results/ai_v2/ai_v2_reranker.joblib"
)

OUTPUT_DIR = "results/ai_v2"

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "ai_v2_localization_results.csv"
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


# ============================================================
# DISTANCE
# ============================================================

def euclidean_distance(x1, y1, x2, y2):

    return float(
        np.sqrt(
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DRIFT-SENSE AI-V2 PHASE 6.4")
    print("AI RELATIVE CANDIDATE LOCALIZATION")
    print("=" * 80)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]

    model_features = bundle.get(
        "features",
        FEATURES
    )

    print()
    print("Model loaded successfully.")
    print("Features :", len(model_features))

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_FILE)

    print()
    print("Dataset rows :", len(df))

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    missing = [
        f
        for f in model_features
        if f not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing features:\n"
            +
            "\n".join(missing)
        )

    # --------------------------------------------------------
    # PREDICT ALL CANDIDATES
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        df[model_features]
    )[:, 1]

    df["ai_probability"] = probabilities

    # --------------------------------------------------------
    # GROUP CANDIDATES
    # --------------------------------------------------------

    group_columns = [
        "sample",
        "noise_level"
    ]

    groups = df.groupby(
        group_columns,
        sort=True
    )

    print()
    print(
        "Candidate groups :",
        groups.ngroups
    )

    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    results = []

    for (sample, noise_level), group in groups:

        group = group.copy()

        # ----------------------------------------------------
        # AI WINNER
        # ----------------------------------------------------

        ai_idx = group[
            "ai_probability"
        ].idxmax()

        ai_row = group.loc[ai_idx]

        ai_x = float(
            ai_row["candidate_x"]
        )

        ai_y = float(
            ai_row["candidate_y"]
        )

        gt_x = float(
            ai_row["gt_x"]
        )

        gt_y = float(
            ai_row["gt_y"]
        )

        ai_error = euclidean_distance(
            ai_x,
            ai_y,
            gt_x,
            gt_y
        )

        # ----------------------------------------------------
        # BASELINE WINNER
        #
        # Original template matching winner
        # = highest template score
        # ----------------------------------------------------

        baseline_idx = group[
            "template_score"
        ].idxmax()

        baseline_row = group.loc[
            baseline_idx
        ]

        baseline_x = float(
            baseline_row["candidate_x"]
        )

        baseline_y = float(
            baseline_row["candidate_y"]
        )

        baseline_error = euclidean_distance(
            baseline_x,
            baseline_y,
            gt_x,
            gt_y
        )

        # ----------------------------------------------------
        # RANKS
        # ----------------------------------------------------

        ai_rank = int(
            ai_row["rank"]
        )

        baseline_rank = int(
            baseline_row["rank"]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        improvement = (
            baseline_error
            -
            ai_error
        )

        if improvement > 1e-9:

            status = "improved"

        elif improvement < -1e-9:

            status = "worse"

        else:

            status = "equal"

        results.append({

            "sample": sample,

            "noise_level": noise_level,

            "gt_x": gt_x,

            "gt_y": gt_y,

            "ai_x": ai_x,

            "ai_y": ai_y,

            "baseline_x": baseline_x,

            "baseline_y": baseline_y,

            "ai_rank": ai_rank,

            "baseline_rank": baseline_rank,

            "ai_probability": float(
                ai_row["ai_probability"]
            ),

            "ai_error": ai_error,

            "baseline_error": baseline_error,

            "improvement": improvement,

            "status": status,

        })

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("AI-V2 LOCALIZATION RESULTS")
    print("=" * 80)

    print()

    print(
        "Evaluated cases :",
        len(results_df)
    )

    print()

    print(
        "AI mean error :",
        f"{results_df['ai_error'].mean():.3f}",
        "px"
    )

    print(
        "AI median error :",
        f"{results_df['ai_error'].median():.3f}",
        "px"
    )

    print(
        "AI worst error :",
        f"{results_df['ai_error'].max():.3f}",
        "px"
    )

    print()

    print(
        "Baseline mean error :",
        f"{results_df['baseline_error'].mean():.3f}",
        "px"
    )

    print(
        "Baseline median error :",
        f"{results_df['baseline_error'].median():.3f}",
        "px"
    )

    print(
        "Baseline worst error :",
        f"{results_df['baseline_error'].max():.3f}",
        "px"
    )

    # --------------------------------------------------------
    # PASS RATES
    # --------------------------------------------------------

    print()
    print("PASS RATES")
    print("-" * 80)

    thresholds = [
        1,
        2,
        5,
        20,
        50
    ]

    for threshold in thresholds:

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
            f"@ {threshold:2d}px : "
            f"AI {ai_pass}/{total} "
            f"({100 * ai_pass / total:.2f}%)"
            f"   |   "
            f"Baseline {baseline_pass}/{total} "
            f"({100 * baseline_pass / total:.2f}%)"
        )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    print()
    print("AI-V2 IMPROVEMENT")
    print("-" * 80)

    improved = (
        results_df["status"]
        == "improved"
    ).sum()

    worse = (
        results_df["status"]
        == "worse"
    ).sum()

    equal = (
        results_df["status"]
        == "equal"
    ).sum()

    print(
        "Improved :",
        f"{improved}/{len(results_df)}"
    )

    print(
        "Worse    :",
        f"{worse}/{len(results_df)}"
    )

    print(
        "Equal    :",
        f"{equal}/{len(results_df)}"
    )

    # --------------------------------------------------------
    # BEST IMPROVEMENTS
    # --------------------------------------------------------

    print()
    print("BEST AI-V2 IMPROVEMENTS")
    print("-" * 80)

    best = results_df.sort_values(
        "improvement",
        ascending=False
    ).head(10)

    print(
        best[
            [
                "sample",
                "noise_level",
                "ai_rank",
                "baseline_rank",
                "ai_error",
                "baseline_error",
                "improvement",
                "ai_probability",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # WORST AI-V2 CASES
    # --------------------------------------------------------

    print()
    print("WORST AI-V2 CASES")
    print("-" * 80)

    worst = results_df.sort_values(
        "ai_error",
        ascending=False
    ).head(10)

    print(
        worst[
            [
                "sample",
                "noise_level",
                "ai_rank",
                "baseline_rank",
                "ai_error",
                "baseline_error",
                "ai_probability",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # HARD CASE ANALYSIS
    # --------------------------------------------------------

    hard_samples = [
        "sample_009",
        "sample_013",
        "sample_023",
        "sample_026",
    ]

    hard = results_df[
        results_df["sample"].isin(
            hard_samples
        )
    ].copy()

    print()
    print("=" * 80)
    print("AI-V2 HARD CASE ANALYSIS")
    print("=" * 80)

    if len(hard) > 0:

        print(
            hard[
                [
                    "sample",
                    "noise_level",
                    "ai_rank",
                    "baseline_rank",
                    "ai_x",
                    "ai_y",
                    "baseline_x",
                    "baseline_y",
                    "ai_error",
                    "baseline_error",
                    "improvement",
                    "ai_probability",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # BY NOISE LEVEL
    # --------------------------------------------------------

    print()
    print("RESULTS BY NOISE LEVEL")
    print("-" * 80)

    noise_summary = (
        results_df
        .groupby("noise_level")
        .agg(
            samples=("sample", "count"),

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
            ),
        )
    )

    print(
        noise_summary.to_string()
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("AI-V2 LOCALIZATION EVALUATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Saved to:",
        os.path.abspath(
            RESULTS_FILE
        )
    )


if __name__ == "__main__":
    main()