import os
import numpy as np
import pandas as pd


# ============================================================
# DRIFT-SENSE AI-V2
# RELATIVE CANDIDATE DATASET BUILDER
# ============================================================

INPUT_FILE = "results/ai/candidate_dataset.csv"
OUTPUT_DIR = "results/ai_v2"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_ranking_dataset.csv"
)


# Features used for relative comparison
BASE_FEATURES = [
    "template_score",
    "gray_score",
    "edge_score",
    "gradient_score",
    "structural_score",
    "center_distance",
]


def main():

    print("=" * 75)
    print("DRIFT-SENSE AI-V2 PHASE")
    print("RELATIVE CANDIDATE RANKING DATASET")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load existing candidate dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print()
    print(f"Input rows : {len(df)}")

    print()
    print("Input columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "sample",
        "noise_level",
        "rank",
        "candidate_x",
        "candidate_y",
        "gt_x",
        "gt_y",
        "template_score",
        "gray_score",
        "edge_score",
        "gradient_score",
        "structural_score",
        "center_distance",
        "distance_to_gt",
        "label",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # 3. Define candidate groups
    #
    # Each image/noise combination is one ranking problem.
    #
    # Example:
    #
    # sample_013 + clean
    #       |
    #       +-- candidate 1
    #       +-- candidate 2
    #       +-- candidate 3
    #       ...
    #       +-- candidate 30
    #
    # --------------------------------------------------------

    group_columns = [
        "sample",
        "noise_level"
    ]

    groups = df.groupby(
        group_columns,
        sort=False
    )

    output_rows = []

    total_groups = len(groups)

    print()
    print(f"Candidate groups : {total_groups}")

    # --------------------------------------------------------
    # 4. Process every candidate group
    # --------------------------------------------------------

    for group_index, (
        group_key,
        group
    ) in enumerate(
        groups,
        start=1
    ):

        group = group.copy()

        sample = group_key[0]
        noise_level = group_key[1]

        # ----------------------------------------------------
        # Reference / group statistics
        # ----------------------------------------------------

        # Best value for each score
        best_template = group[
            "template_score"
        ].max()

        best_gray = group[
            "gray_score"
        ].max()

        best_edge = group[
            "edge_score"
        ].max()

        best_gradient = group[
            "gradient_score"
        ].max()

        best_structural = group[
            "structural_score"
        ].max()

        # Minimum center distance
        best_center_distance = group[
            "center_distance"
        ].min()

        # Maximum rank available
        max_rank = group[
            "rank"
        ].max()

        # ----------------------------------------------------
        # Candidate spatial statistics
        # ----------------------------------------------------

        candidate_x_values = group[
            "candidate_x"
        ].values

        candidate_y_values = group[
            "candidate_y"
        ].values

        # ----------------------------------------------------
        # Process each candidate
        # ----------------------------------------------------

        for _, row in group.iterrows():

            candidate_x = row[
                "candidate_x"
            ]

            candidate_y = row[
                "candidate_y"
            ]

            # -----------------------------------------------
            # Relative score features
            # -----------------------------------------------

            template_gap = (
                best_template
                - row["template_score"]
            )

            gray_gap = (
                best_gray
                - row["gray_score"]
            )

            edge_gap = (
                best_edge
                - row["edge_score"]
            )

            gradient_gap = (
                best_gradient
                - row["gradient_score"]
            )

            structural_gap = (
                best_structural
                - row["structural_score"]
            )

            # -----------------------------------------------
            # Normalized rank
            # -----------------------------------------------

            if max_rank > 1:

                normalized_rank = (
                    row["rank"] - 1
                ) / (
                    max_rank - 1
                )

            else:

                normalized_rank = 0.0

            # -----------------------------------------------
            # Distance from the strongest template candidate
            # -----------------------------------------------

            best_template_index = (
                group[
                    "template_score"
                ].idxmax()
            )

            best_template_x = group.loc[
                best_template_index,
                "candidate_x"
            ]

            best_template_y = group.loc[
                best_template_index,
                "candidate_y"
            ]

            distance_from_template_best = np.sqrt(
                (
                    candidate_x
                    - best_template_x
                ) ** 2
                +
                (
                    candidate_y
                    - best_template_y
                ) ** 2
            )

            # -----------------------------------------------
            # Candidate neighborhood density
            #
            # Number of other candidates close to this
            # candidate.
            # -----------------------------------------------

            dx = (
                candidate_x
                - candidate_x_values
            )

            dy = (
                candidate_y
                - candidate_y_values
            )

            distances = np.sqrt(
                dx ** 2 + dy ** 2
            )

            neighborhood_density_50 = np.sum(
                (
                    distances > 0
                )
                &
                (
                    distances <= 50
                )
            )

            neighborhood_density_100 = np.sum(
                (
                    distances > 0
                )
                &
                (
                    distances <= 100
                )
            )

            # -----------------------------------------------
            # Distance to closest competing candidate
            # -----------------------------------------------

            other_distances = distances[
                distances > 0
            ]

            if len(other_distances) > 0:

                nearest_candidate_distance = (
                    np.min(other_distances)
                )

            else:

                nearest_candidate_distance = 0.0

            # -----------------------------------------------
            # Combined classical score
            #
            # This gives AI-V2 a representation of the
            # classical evidence without replacing it.
            # -----------------------------------------------

            combined_score = (
                0.30 * row["template_score"]
                +
                0.15 * row["gray_score"]
                +
                0.15 * row["edge_score"]
                +
                0.20 * row["gradient_score"]
                +
                0.20 * row["structural_score"]
            )

            # -----------------------------------------------
            # Relative combined score
            # -----------------------------------------------

            combined_scores = (
                0.30 * group["template_score"]
                +
                0.15 * group["gray_score"]
                +
                0.15 * group["edge_score"]
                +
                0.20 * group["gradient_score"]
                +
                0.20 * group["structural_score"]
            )

            best_combined_score = (
                combined_scores.max()
            )

            combined_gap = (
                best_combined_score
                - combined_score
            )

            # -----------------------------------------------
            # Store row
            # -----------------------------------------------

            output_rows.append({

                # Group identity
                "sample": sample,
                "noise_level": noise_level,

                # Original candidate information
                "rank": row["rank"],
                "candidate_x": candidate_x,
                "candidate_y": candidate_y,

                # Ground truth
                "gt_x": row["gt_x"],
                "gt_y": row["gt_y"],

                # Original features
                "template_score": row[
                    "template_score"
                ],

                "gray_score": row[
                    "gray_score"
                ],

                "edge_score": row[
                    "edge_score"
                ],

                "gradient_score": row[
                    "gradient_score"
                ],

                "structural_score": row[
                    "structural_score"
                ],

                "center_distance": row[
                    "center_distance"
                ],

                # ------------------------------------------------
                # NEW AI-V2 RELATIVE FEATURES
                # ------------------------------------------------

                "template_gap": template_gap,

                "gray_gap": gray_gap,

                "edge_gap": edge_gap,

                "gradient_gap": gradient_gap,

                "structural_gap": structural_gap,

                "combined_score": combined_score,

                "combined_gap": combined_gap,

                "normalized_rank": normalized_rank,

                "distance_from_template_best":
                    distance_from_template_best,

                "nearest_candidate_distance":
                    nearest_candidate_distance,

                "neighborhood_density_50":
                    neighborhood_density_50,

                "neighborhood_density_100":
                    neighborhood_density_100,

                # Original ground-truth information
                # Used ONLY for training/evaluation.
                "distance_to_gt": row[
                    "distance_to_gt"
                ],

                "label": row[
                    "label"
                ],
            })

        if (
            group_index % 10 == 0
            or group_index == total_groups
        ):

            print(
                f"[OK] {group_index}/{total_groups} "
                f"groups processed"
            )

    # --------------------------------------------------------
    # 5. Create DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # 6. Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 7. Save dataset
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 8. Statistics
    # --------------------------------------------------------

    positive_count = int(
        result["label"].sum()
    )

    negative_count = (
        len(result)
        - positive_count
    )

    print()
    print("=" * 75)
    print("AI-V2 DATASET COMPLETE")
    print("=" * 75)

    print(
        f"Rows generated : {len(result)}"
    )

    print(
        f"Positive rows  : {positive_count}"
    )

    print(
        f"Negative rows  : {negative_count}"
    )

    print()

    print(
        "Candidate groups : "
        f"{result.groupby(group_columns).ngroups}"
    )

    print()

    print("AI-V2 FEATURES")
    print("-" * 75)

    feature_columns = [
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

    for feature in feature_columns:

        print(
            f"  {feature}"
        )

    print()
    print(
        f"Saved to: {os.path.abspath(OUTPUT_FILE)}"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()