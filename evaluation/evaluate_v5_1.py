import os
import json
import re
import subprocess
import time
import math


DATA_DIR = "data"
TOTAL_SAMPLES = 30


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def get_ground_truth(metadata_path):

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return (
        float(metadata["gt_x"]),
        float(metadata["gt_y"])
    )


def run_v5_1(reference, search):

    command = [
        "python",
        "localization\\baseline_v5_1.py",
        reference,
        search
    ]

    start = time.perf_counter()

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    runtime = (
        time.perf_counter() - start
    ) * 1000.0

    output = result.stdout

    # Look for:
    # Final predicted center : (341.00, 824.00)

    match = re.search(
        r"Final predicted center\s*:\s*"
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*\)",
        output
    )

    if not match:

        return None, runtime, output

    x = float(match.group(1))
    y = float(match.group(2))

    return (
        (x, y),
        runtime,
        output
    )


def main():

    print()
    print("=" * 75)
    print("DRIFT-SENSE V5.1 DATASET EVALUATION")
    print("=" * 75)

    results = []

    for i in range(
        1,
        TOTAL_SAMPLES + 1
    ):

        sample_name = (
            f"sample_{i:03d}"
        )

        sample_dir = os.path.join(
            DATA_DIR,
            sample_name
        )

        reference = os.path.join(
            sample_dir,
            "reference.png"
        )

        search = os.path.join(
            sample_dir,
            "search.png"
        )

        metadata = os.path.join(
            sample_dir,
            "metadata.json"
        )

        # -----------------------------------------------------
        # Check required files
        # -----------------------------------------------------

        if not all(
            os.path.exists(path)
            for path in [
                reference,
                search,
                metadata
            ]
        ):

            print(
                f"[SKIP] {sample_name} "
                f"- missing required file"
            )

            continue

        # -----------------------------------------------------
        # Ground truth
        # -----------------------------------------------------

        try:

            gt_x, gt_y = get_ground_truth(
                metadata
            )

        except Exception as e:

            print(
                f"[SKIP] {sample_name} "
                f"- metadata error: {e}"
            )

            continue

        # -----------------------------------------------------
        # V5.1
        # -----------------------------------------------------

        prediction, runtime, output = run_v5_1(
            reference,
            search
        )

        if prediction is None:

            print(
                f"[FAILED] {sample_name} "
                f"- V5.1 prediction unavailable"
            )

            continue

        pred_x, pred_y = prediction

        error = distance(
            pred_x,
            pred_y,
            gt_x,
            gt_y
        )

        results.append(
            {
                "sample": sample_name,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "error": error,
                "runtime_ms": runtime
            }
        )

        print(
            f"{sample_name}: "
            f"V5.1={error:.2f}px"
        )

    # ---------------------------------------------------------
    # No results
    # ---------------------------------------------------------

    if not results:

        print()
        print("No samples were successfully evaluated.")
        return

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    errors = [
        r["error"]
        for r in results
    ]

    runtimes = [
        r["runtime_ms"]
        for r in results
    ]

    errors_sorted = sorted(errors)

    mean_error = sum(errors) / len(errors)

    median_error = (
        errors_sorted[len(errors) // 2]
        if len(errors) % 2 == 1
        else
        (
            errors_sorted[len(errors) // 2 - 1]
            +
            errors_sorted[len(errors) // 2]
        ) / 2
    )

    worst_error = max(errors)

    mean_runtime = (
        sum(runtimes) /
        len(runtimes)
    )

    print()
    print("=" * 75)
    print("V5.1 RESULTS")
    print("=" * 75)

    print(
        f"Evaluated samples : "
        f"{len(results)}"
    )

    print()

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
        f"{mean_runtime:.3f} ms"
    )

    print()

    # ---------------------------------------------------------
    # Pass rates
    # ---------------------------------------------------------

    for threshold in [
        5,
        4,
        2,
        1
    ]:

        passed = sum(
            error <= threshold
            for error in errors
        )

        rate = (
            passed /
            len(errors)
        ) * 100.0

        print(
            f"Pass rate @ {threshold}px : "
            f"{passed}/{len(errors)} "
            f"({rate:.2f}%)"
        )

    print("=" * 75)

    # ---------------------------------------------------------
    # Worst samples
    # ---------------------------------------------------------

    print()
    print("HARDEST SAMPLES")
    print("-" * 75)

    hardest = sorted(
        results,
        key=lambda r: r["error"],
        reverse=True
    )

    for result in hardest[:5]:

        print(
            f"{result['sample']}: "
            f"{result['error']:.2f} px"
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    csv_path = (
        "results/"
        "v5_1_evaluation_results.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "sample,pred_x,pred_y,"
            "gt_x,gt_y,error,runtime_ms\n"
        )

        for r in results:

            f.write(
                f"{r['sample']},"
                f"{r['pred_x']:.4f},"
                f"{r['pred_y']:.4f},"
                f"{r['gt_x']:.4f},"
                f"{r['gt_y']:.4f},"
                f"{r['error']:.4f},"
                f"{r['runtime_ms']:.4f}\n"
            )

    print()
    print(
        f"CSV saved to: {csv_path}"
    )


if __name__ == "__main__":
    main()