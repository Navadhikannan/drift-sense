from pathlib import Path
import sys
import csv
import time
import json
import cv2
import numpy as np

# ============================================================
# DRIFT-SENSE PHASE 6
# ROBUSTNESS EVALUATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ROBUSTNESS_DIR = BASE_DIR / "data" / "robustness"
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR))

from localization.baseline_v5_1 import run_v5_1


NOISE_LEVELS = [
    "clean",
    "low",
    "medium",
    "high",
]

THRESHOLDS = [5.0, 4.0, 2.0, 1.0]


def load_ground_truth(metadata_path):
    """Read ground-truth center from metadata.json."""

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    gt_x = float(metadata["gt_x"])
    gt_y = float(metadata["gt_y"])

    return gt_x, gt_y


def euclidean_error(pred_x, pred_y, gt_x, gt_y):
    return float(
        np.sqrt(
            (pred_x - gt_x) ** 2 +
            (pred_y - gt_y) ** 2
        )
    )


def evaluate_sample(reference_path, search_path, metadata_path):

    gt_x, gt_y = load_ground_truth(metadata_path)

    start = time.perf_counter()

    result = run_v5_1(
        str(reference_path),
        str(search_path)
    )

    runtime = time.perf_counter() - start

    # V5.1 is expected to return:
    # x, y, score
    pred_x = float(result["x"])
    pred_y = float(result["y"])
    score = float(result["score"])

    error = euclidean_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    return {
        "gt_x": gt_x,
        "gt_y": gt_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "error": error,
        "score": score,
        "runtime_sec": runtime,
    }


def main():

    print("=" * 70)
    print("DRIFT-SENSE PHASE 6 - ROBUSTNESS EVALUATION")
    print("=" * 70)

    if not ROBUSTNESS_DIR.exists():
        print("[ERROR] Robustness dataset not found:")
        print(ROBUSTNESS_DIR)
        return

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    for level in NOISE_LEVELS:

        level_dir = ROBUSTNESS_DIR / level

        if not level_dir.exists():
            print(f"[SKIP] Missing level: {level}")
            continue

        print()
        print("-" * 70)
        print(f"NOISE LEVEL: {level.upper()}")
        print("-" * 70)

        sample_dirs = sorted(
            [
                p for p in level_dir.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            ]
        )

        for sample_dir in sample_dirs:

            reference_path = sample_dir / "reference.png"
            search_path = sample_dir / "search.png"
            metadata_path = sample_dir / "metadata.json"

            if not (
                reference_path.exists()
                and search_path.exists()
                and metadata_path.exists()
            ):
                print(
                    f"[SKIP] {level}/{sample_dir.name} "
                    "- missing required file"
                )
                continue

            try:

                result = evaluate_sample(
                    reference_path,
                    search_path,
                    metadata_path
                )

                result["noise_level"] = level
                result["sample"] = sample_dir.name

                all_results.append(result)

                print(
                    f"{sample_dir.name}: "
                    f"error={result['error']:.2f}px | "
                    f"score={result['score']:.4f} | "
                    f"runtime={result['runtime_sec'] * 1000:.2f}ms"
                )

            except Exception as exc:

                print(
                    f"[ERROR] {level}/{sample_dir.name}: "
                    f"{exc}"
                )


    # ========================================================
    # CSV
    # ========================================================

    csv_path = RESULTS_DIR / "robustness_noise_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_level",
                "sample",
                "gt_x",
                "gt_y",
                "pred_x",
                "pred_y",
                "error",
                "score",
                "runtime_sec",
            ]
        )

        writer.writeheader()

        for row in all_results:
            writer.writerow(row)


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    for level in NOISE_LEVELS:

        rows = [
            r for r in all_results
            if r["noise_level"] == level
        ]

        if not rows:
            continue

        errors = np.array(
            [r["error"] for r in rows],
            dtype=float
        )

        runtimes = np.array(
            [r["runtime_sec"] for r in rows],
            dtype=float
        )

        print()
        print(f"## {level.upper()}")

        print(
            f"Samples           : {len(rows)}"
        )

        print(
            f"Mean error        : "
            f"{np.mean(errors):.3f} px"
        )

        print(
            f"Median error      : "
            f"{np.median(errors):.3f} px"
        )

        print(
            f"Worst-case error  : "
            f"{np.max(errors):.3f} px"
        )

        print(
            f"Mean runtime      : "
            f"{np.mean(runtimes) * 1000:.3f} ms"
        )

        for threshold in THRESHOLDS:

            passed = int(
                np.sum(errors <= threshold)
            )

            total = len(errors)

            rate = (
                100.0 * passed / total
                if total > 0
                else 0.0
            )

            print(
                f"Pass rate @ "
                f"{threshold:g}px : "
                f"{passed}/{total} "
                f"({rate:.2f}%)"
            )


    print()
    print("=" * 70)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Evaluated samples : {len(all_results)}")
    print(f"CSV saved to      : {csv_path}")


if __name__ == "__main__":
    main()