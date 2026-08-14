import os
import pandas as pd
import numpy as np


INPUT_FILE = "results/ai_v2/ai_v2_test_predictions.csv"
OUTPUT_DIR = "results/ai_v2/confidence_analysis"


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 80)
print("DRIFT-SENSE PHASE 6.5")
print("AI CONFIDENCE ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# 1. Load predictions
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print()
print(f"Prediction rows : {len(df)}")
print(f"Input file      : {INPUT_FILE}")


# ------------------------------------------------------------
# 2. Inspect columns
# ------------------------------------------------------------

print()
print("Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Calculate confidence statistics per candidate group
# ------------------------------------------------------------

group_columns = [
    "sample",
    "noise_level"
]

required_columns = [
    "sample",
    "noise_level",
    "probability"
]

for column in required_columns:

    if column not in df.columns:
        raise KeyError(
            f"Required column '{column}' "
            f"not found in prediction file."
        )


records = []


for (sample, noise), group in df.groupby(group_columns):

    group = group.sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    # Best candidate
    best = group.iloc[0]

    # Second-best candidate
    if len(group) > 1:
        second = group.iloc[1]
        second_probability = float(
            second["probability"]
        )
    else:
        second_probability = 0.0

    best_probability = float(
        best["probability"]
    )

    confidence_gap = (
        best_probability
        - second_probability
    )

    # --------------------------------------------------------
    # Confidence classification
    # --------------------------------------------------------

    if best_probability >= 0.90 and confidence_gap >= 0.10:

        confidence = "HIGH"

    elif best_probability >= 0.70 and confidence_gap >= 0.05:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"

    record = {
        "sample": sample,
        "noise_level": noise,
        "best_rank": int(best["rank"]),
        "best_x": float(best["candidate_x"]),
        "best_y": float(best["candidate_y"]),
        "best_probability": best_probability,
        "second_probability": second_probability,
        "confidence_gap": confidence_gap,
        "confidence": confidence,
    }

    # Add error if available
    if "distance_to_gt" in best.index:

        record["error"] = float(
            best["distance_to_gt"]
        )

    elif "ai_error" in best.index:

        record["error"] = float(
            best["ai_error"]
        )

    else:

        record["error"] = np.nan

    records.append(record)


confidence_df = pd.DataFrame(records)


# ------------------------------------------------------------
# 4. Save complete confidence table
# ------------------------------------------------------------

confidence_file = os.path.join(
    OUTPUT_DIR,
    "confidence_results.csv"
)

confidence_df.to_csv(
    confidence_file,
    index=False
)


# ------------------------------------------------------------
# 5. Confidence distribution
# ------------------------------------------------------------

print()
print("=" * 80)
print("CONFIDENCE DISTRIBUTION")
print("=" * 80)

print(
    confidence_df["confidence"]
    .value_counts()
)


# ------------------------------------------------------------
# 6. Confidence vs localization error
# ------------------------------------------------------------

print()
print("=" * 80)
print("CONFIDENCE VS LOCALIZATION ERROR")
print("=" * 80)

summary = (
    confidence_df
    .groupby("confidence")
    .agg(
        samples=("sample", "count"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        worst_error=("error", "max"),
        mean_probability=("best_probability", "mean"),
        mean_gap=("confidence_gap", "mean")
    )
    .sort_index()
)

print(summary)


summary_file = os.path.join(
    OUTPUT_DIR,
    "confidence_summary.csv"
)

summary.to_csv(summary_file)


# ------------------------------------------------------------
# 7. Low-confidence cases
# ------------------------------------------------------------

low_confidence = confidence_df[
    confidence_df["confidence"] == "LOW"
].sort_values(
    "error",
    ascending=False
)


low_file = os.path.join(
    OUTPUT_DIR,
    "low_confidence_cases.csv"
)

low_confidence.to_csv(
    low_file,
    index=False
)


# ------------------------------------------------------------
# 8. High-error cases
# ------------------------------------------------------------

high_error = confidence_df[
    confidence_df["error"] > 5
].sort_values(
    "error",
    ascending=False
)


high_error_file = os.path.join(
    OUTPUT_DIR,
    "high_error_cases.csv"
)

high_error.to_csv(
    high_error_file,
    index=False
)


# ------------------------------------------------------------
# 9. Print worst cases
# ------------------------------------------------------------

print()
print("=" * 80)
print("WORST LOCALIZATION CASES")
print("=" * 80)

columns = [
    "sample",
    "noise_level",
    "best_rank",
    "best_probability",
    "second_probability",
    "confidence_gap",
    "confidence",
    "error"
]

print(
    confidence_df
    .sort_values(
        "error",
        ascending=False
    )
    .head(15)[columns]
    .to_string(index=False)
)


# ------------------------------------------------------------
# 10. Print low-confidence cases
# ------------------------------------------------------------

print()
print("=" * 80)
print("LOW-CONFIDENCE CASES")
print("=" * 80)

if len(low_confidence) == 0:

    print("No low-confidence cases found.")

else:

    print(
        low_confidence[
            columns
        ]
        .head(20)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 11. Final output
# ------------------------------------------------------------

print()
print("=" * 80)
print("PHASE 6.5 COMPLETE")
print("=" * 80)

print()
print("Generated:")

print(
    f"  {confidence_file}"
)

print(
    f"  {summary_file}"
)

print(
    f"  {low_file}"
)

print(
    f"  {high_error_file}"
)

print()
print("Next step:")
print("Run the final benchmark.")