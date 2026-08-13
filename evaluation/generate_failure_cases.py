import os
import json
import cv2
import pandas as pd


CSV_PATH = "results/evaluation_results.csv"
DATA_DIR = "data"
OUTPUT_DIR = "results/failure_cases"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load evaluation results
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Four hardest V1 samples
hardest = (
    df.sort_values(
        "v1_error",
        ascending=False
    )
    .head(4)
)

print("=" * 70)
print("DRIFT-SENSE FAILURE CASE VISUALIZATION")
print("=" * 70)

print()

print("Selected failure cases:")

for _, row in hardest.iterrows():

    print(
        f"{row['sample']} : "
        f"{row['v1_error']:.2f} px"
    )


# ---------------------------------------------------------
# Generate visualization for each failure case
# ---------------------------------------------------------

for _, row in hardest.iterrows():

    sample = row["sample"]

    sample_dir = os.path.join(
        DATA_DIR,
        sample
    )

    search_path = os.path.join(
        sample_dir,
        "search.png"
    )

    if not os.path.exists(search_path):

        print(
            f"[SKIP] {sample} - search image missing"
        )

        continue

    search = cv2.imread(
        search_path
    )

    if search is None:

        print(
            f"[SKIP] {sample} - could not read image"
        )

        continue

    # -----------------------------------------------------
    # Coordinates
    # -----------------------------------------------------

    gt_x = float(row["gt_x"])
    gt_y = float(row["gt_y"])

    pred_x = float(row["v1_x"])
    pred_y = float(row["v1_y"])

    error = float(row["v1_error"])

    # -----------------------------------------------------
    # Draw ground truth
    # -----------------------------------------------------

    gt_point = (
        int(round(gt_x)),
        int(round(gt_y))
    )

    pred_point = (
        int(round(pred_x)),
        int(round(pred_y))
    )

    # Ground-truth circle
    cv2.circle(
        search,
        gt_point,
        12,
        (0, 0, 255),
        3
    )

    # Predicted circle
    cv2.circle(
        search,
        pred_point,
        12,
        (255, 0, 0),
        3
    )

    # -----------------------------------------------------
    # Connect GT and prediction
    # -----------------------------------------------------

    cv2.line(
        search,
        gt_point,
        pred_point,
        (0, 255, 255),
        2
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    cv2.putText(
        search,
        "GT",
        (
            gt_point[0] + 15,
            gt_point[1] - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        search,
        "V1",
        (
            pred_point[0] + 15,
            pred_point[1] + 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    # -----------------------------------------------------
    # Information box
    # -----------------------------------------------------

    cv2.rectangle(
        search,
        (15, 15),
        (390, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        search,
        f"Sample: {sample}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        search,
        f"GT: ({gt_x:.1f}, {gt_y:.1f})",
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        search,
        f"V1: ({pred_x:.1f}, {pred_y:.1f})",
        (30, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        search,
        f"Error: {error:.2f} px",
        (30, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{sample}_failure.png"
    )

    cv2.imwrite(
        output_path,
        search
    )

    print(
        f"[OK] {output_path}"
    )


print()
print("=" * 70)
print("FAILURE VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"Saved to: {OUTPUT_DIR}"
)