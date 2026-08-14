from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# DRIFT-SENSE PHASE 6.1
# ROBUSTNESS FAILURE ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "robustness_noise_results.csv"
OUTPUT_DIR = BASE_DIR / "results" / "robustness_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 70)
    print("DRIFT-SENSE PHASE 6.1 - FAILURE ANALYSIS")
    print("=" * 70)

    if not CSV_PATH.exists():
        print("[ERROR] Results CSV not found:")
        print(CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    print()
    print(f"Total evaluated cases : {len(df)}")
    print(f"Noise levels          : {df['noise_level'].nunique()}")
    print(f"Samples per level     :")
    print(df["noise_level"].value_counts().sort_index())

    # --------------------------------------------------------
    # Failure categories
    # --------------------------------------------------------

    thresholds = [1, 2, 5, 20, 50]

    print()
    print("=" * 70)
    print("FAILURE COUNTS")
    print("=" * 70)

    for threshold in thresholds:

        failures = df[df["error"] > threshold]

        print(
            f"Error > {threshold:>2}px : "
            f"{len(failures)}/{len(df)} "
            f"({100 * len(failures) / len(df):.2f}%)"
        )

    # --------------------------------------------------------
    # Statistics by noise level
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR BY NOISE LEVEL")
    print("=" * 70)

    summary = (
        df.groupby("noise_level")["error"]
        .agg(
            samples="count",
            mean_error="mean",
            median_error="median",
            worst_error="max"
        )
        .reindex(
            ["clean", "low", "medium", "high"]
        )
    )

    summary["pass_5px"] = (
        df.groupby("noise_level")["error"]
        .apply(lambda x: (x <= 5).sum())
        .reindex(
            ["clean", "low", "medium", "high"]
        )
    )

    summary["pass_5px_rate"] = (
        summary["pass_5px"]
        / summary["samples"]
        * 100
    )

    print(summary.round(3).to_string())

    # --------------------------------------------------------
    # Worst cases
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 10 WORST CASES")
    print("=" * 70)

    worst = (
        df.sort_values(
            "error",
            ascending=False
        )
        .head(10)
    )

    print(
        worst[
            [
                "noise_level",
                "sample",
                "error",
                "score",
                "runtime_sec"
            ]
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Catastrophic failures
    # --------------------------------------------------------

    catastrophic = df[
        df["error"] > 50
    ].sort_values(
        "error",
        ascending=False
    )

    print()
    print("=" * 70)
    print("CATASTROPHIC FAILURES (>50 px)")
    print("=" * 70)

    if len(catastrophic) == 0:

        print("None")

    else:

        print(
            catastrophic[
                [
                    "noise_level",
                    "sample",
                    "gt_x",
                    "gt_y",
                    "pred_x",
                    "pred_y",
                    "error",
                    "score"
                ]
            ]
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # High-score failures
    # --------------------------------------------------------

    high_score_failures = df[
        (df["score"] >= 0.85)
        &
        (df["error"] > 5)
    ].sort_values(
        "error",
        ascending=False
    )

    print()
    print("=" * 70)
    print("HIGH-SCORE BUT WRONG (>5 px error, score >= 0.85)")
    print("=" * 70)

    if len(high_score_failures) == 0:

        print("None")

    else:

        print(
            high_score_failures[
                [
                    "noise_level",
                    "sample",
                    "error",
                    "score"
                ]
            ]
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNTIME BY NOISE LEVEL")
    print("=" * 70)

    runtime_summary = (
        df.groupby("noise_level")["runtime_sec"]
        .agg(
            mean_runtime_ms=lambda x: np.mean(x) * 1000,
            median_runtime_ms=lambda x: np.median(x) * 1000,
            worst_runtime_ms=lambda x: np.max(x) * 1000
        )
        .reindex(
            ["clean", "low", "medium", "high"]
        )
    )

    print(
        runtime_summary.round(3).to_string()
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR /
        "noise_summary.csv"
    )

    worst_path = (
        OUTPUT_DIR /
        "worst_cases.csv"
    )

    catastrophic_path = (
        OUTPUT_DIR /
        "catastrophic_failures.csv"
    )

    high_score_path = (
        OUTPUT_DIR /
        "high_score_failures.csv"
    )

    summary.to_csv(
        summary_path
    )

    worst.to_csv(
        worst_path,
        index=False
    )

    catastrophic.to_csv(
        catastrophic_path,
        index=False
    )

    high_score_failures.to_csv(
        high_score_path,
        index=False
    )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 6.1 COMPLETE")
    print("=" * 70)

    print()
    print("Generated:")
    print(f"  {summary_path}")
    print(f"  {worst_path}")
    print(f"  {catastrophic_path}")
    print(f"  {high_score_path}")

    print()
    print("Next step:")
    print("Use these failure cases to design V5.2.")
    print()


if __name__ == "__main__":
    main()