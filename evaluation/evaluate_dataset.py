import os
import sys
import csv
import json
import math
import time
import statistics
import cv2

# ------------------------------------------------------------
# Allow imports from project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from localization.baseline import localize as localize_v1
from localization.baseline_v3 import localize as localize_v3


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_DIR = "data"
OUTPUT_CSV = "results/evaluation_results.csv"

THRESHOLDS = [5.0, 4.0, 2.0, 1.0]


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def calculate_error(pred_x, pred_y, gt_x, gt_y):

    return math.sqrt(
        (pred_x - gt_x) ** 2 +
        (pred_y - gt_y) ** 2
    )


def load_ground_truth(metadata_path):

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return (
        float(metadata["gt_x"]),
        float(metadata["gt_y"])
    )


# ------------------------------------------------------------
# V1 evaluation
# ------------------------------------------------------------

def evaluate_v1(reference_path, search_path, gt_x, gt_y):

    start = time.perf_counter()

    pred_x, pred_y, score = localize_v1(
        reference_path,
        search_path
    )

    elapsed = time.perf_counter() - start

    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    return (
        pred_x,
        pred_y,
        error,
        score,
        elapsed
    )


# ------------------------------------------------------------
# V3 evaluation
# ------------------------------------------------------------

def evaluate_v3(reference_path, search_path, gt_x, gt_y):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(reference_path)

    if search is None:
        raise FileNotFoundError(search_path)

    start = time.perf_counter()

    (
        pred_x,
        pred_y,
        ref_features,
        search_features,
        good_matches
    ) = localize_v3(
        reference,
        search
    )

    elapsed = time.perf_counter() - start

    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    return (
        pred_x,
        pred_y,
        error,
        good_matches,
        elapsed
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 75)
    print("DRIFT-SENSE DATASET EVALUATION")
    print("=" * 75)

    if not os.path.exists(DATASET_DIR):

        print(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

        sys.exit(1)

    sample_dirs = sorted(
        [
            d
            for d in os.listdir(DATASET_DIR)
            if d.startswith("sample_")
            and os.path.isdir(
                os.path.join(DATASET_DIR, d)
            )
        ]
    )

    if not sample_dirs:

        print("No sample folders found.")
        sys.exit(1)

    print(
        f"Samples found : {len(sample_dirs)}"
    )

    print(
        "Thresholds    : "
        "5 px, 4 px, 2 px, 1 px"
    )

    print()

    results = []

    v1_errors = []
    v3_errors = []

    v1_times = []
    v3_times = []

    for sample in sample_dirs:

        sample_path = os.path.join(
            DATASET_DIR,
            sample
        )

        reference_path = os.path.join(
            sample_path,
            "reference.png"
        )

        search_path = os.path.join(
            sample_path,
            "search.png"
        )

        metadata_path = os.path.join(
            sample_path,
            "metadata.json"
        )

        required_files = [
            reference_path,
            search_path,
            metadata_path
        ]

        if not all(
            os.path.exists(path)
            for path in required_files
        ):

            print(
                f"[SKIP] {sample} "
                f"- missing required file"
            )

            continue

        try:

            gt_x, gt_y = load_ground_truth(
                metadata_path
            )

            # =================================================
            # V1
            # =================================================

            (
                v1_x,
                v1_y,
                v1_error,
                v1_score,
                v1_time
            ) = evaluate_v1(
                reference_path,
                search_path,
                gt_x,
                gt_y
            )

            v1_errors.append(v1_error)
            v1_times.append(v1_time)

            # =================================================
            # V3
            # =================================================

            try:

                (
                    v3_x,
                    v3_y,
                    v3_error,
                    v3_matches,
                    v3_time
                ) = evaluate_v3(
                    reference_path,
                    search_path,
                    gt_x,
                    gt_y
                )

                v3_errors.append(v3_error)
                v3_times.append(v3_time)

                v3_status = (
                    f"{v3_error:.2f}px"
                )

            except Exception as error:

                v3_x = None
                v3_y = None
                v3_error = None
                v3_matches = 0
                v3_time = None

                v3_status = "FAILED"

                print(
                    f"[V3 ERROR] "
                    f"{sample}: {error}"
                )

            # =================================================
            # Store result
            # =================================================

            results.append(
                {
                    "sample": sample,

                    "gt_x": gt_x,
                    "gt_y": gt_y,

                    "v1_x": v1_x,
                    "v1_y": v1_y,
                    "v1_error": v1_error,
                    "v1_score": v1_score,
                    "v1_runtime_sec": v1_time,

                    "v3_x": v3_x,
                    "v3_y": v3_y,
                    "v3_error": v3_error,
                    "v3_matches": v3_matches,
                    "v3_runtime_sec": v3_time
                }
            )

            print(
                f"{sample}: "
                f"V1={v1_error:.2f}px | "
                f"V3={v3_status}"
            )

        except Exception as error:

            print(
                f"[ERROR] "
                f"{sample}: {error}"
            )

    # ========================================================
    # Save CSV
    # ========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    if results:

        fieldnames = results[0].keys()

        with open(
            OUTPUT_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

    # ========================================================
    # Summary function
    # ========================================================

    def print_summary(
        name,
        errors,
        runtimes
    ):

        if not errors:
            return

        print()
        print("-" * 75)
        print(name)
        print("-" * 75)

        mean_error = statistics.mean(errors)
        median_error = statistics.median(errors)
        worst_error = max(errors)

        mean_runtime = statistics.mean(
            runtimes
        )

        print(
            f"Mean error       : "
            f"{mean_error:.3f} px"
        )

        print(
            f"Median error     : "
            f"{median_error:.3f} px"
        )

        print(
            f"Worst-case error : "
            f"{worst_error:.3f} px"
        )

        print(
            f"Mean runtime     : "
            f"{mean_runtime * 1000:.3f} ms"
        )

        print()

        total = len(errors)

        for threshold in THRESHOLDS:

            passed = sum(
                error <= threshold
                for error in errors
            )

            rate = (
                passed / total
            ) * 100.0

            print(
                f"Pass rate @ "
                f"{threshold:g}px : "
                f"{passed}/{total} "
                f"({rate:.2f}%)"
            )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        f"Evaluated samples : "
        f"{len(results)}"
    )

    print_summary(
        "V1 - Template Matching",
        v1_errors,
        v1_times
    )

    print_summary(
        "V3 - SIFT",
        v3_errors,
        v3_times
    )

    print()
    print(
        f"CSV saved to: "
        f"{OUTPUT_CSV}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()