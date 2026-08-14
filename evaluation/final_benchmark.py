import os
import pandas as pd
import numpy as np


AI_FILE = "results/ai_v2/ai_v2_localization_results.csv"
CONF_FILE = "results/ai_v2/confidence_analysis/confidence_results.csv"

OUTPUT_DIR = "results/final_benchmark"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 80)
print("DRIFT-SENSE FINAL BENCHMARK")
print("=" * 80)


# ============================================================
# 1. Load AI-V2 localization results
# ============================================================

ai = pd.read_csv(AI_FILE)

print()
print(f"AI-V2 rows : {len(ai)}")
print(f"AI file    : {AI_FILE}")


# ============================================================
# 2. Detect baseline columns
# ============================================================

required = [
    "sample",
    "noise_level",
    "ai_error",
    "baseline_error"
]

for column in required:

    if column not in ai.columns:

        raise KeyError(
            f"Missing required column: {column}"
        )


# ============================================================
# 3. Build benchmark summary
# ============================================================

def metrics(values):

    values = np.asarray(
        values,
        dtype=float
    )

    return {
        "samples": len(values),
        "mean_error_px": np.mean(values),
        "median_error_px": np.median(values),
        "worst_error_px": np.max(values),
        "pass_1px": np.sum(values <= 1),
        "pass_2px": np.sum(values <= 2),
        "pass_5px": np.sum(values <= 5),
        "pass_20px": np.sum(values <= 20),
        "pass_50px": np.sum(values <= 50),
    }


baseline_metrics = metrics(
    ai["baseline_error"]
)

ai_metrics = metrics(
    ai["ai_error"]
)


# ============================================================
# 4. Print main comparison
# ============================================================

print()
print("=" * 80)
print("FINAL LOCALIZATION COMPARISON")
print("=" * 80)

comparison = pd.DataFrame(
    {
        "Baseline_V5.1": baseline_metrics,
        "AI_V2": ai_metrics
    }
)

print(
    comparison.to_string()
)


# ============================================================
# 5. Improvement calculations
# ============================================================

mean_improvement = (
    (
        baseline_metrics["mean_error_px"]
        -
        ai_metrics["mean_error_px"]
    )
    /
    baseline_metrics["mean_error_px"]
    * 100
)


worst_improvement = (
    (
        baseline_metrics["worst_error_px"]
        -
        ai_metrics["worst_error_px"]
    )
    /
    baseline_metrics["worst_error_px"]
    * 100
)


print()
print("=" * 80)
print("AI-V2 IMPROVEMENT")
print("=" * 80)

print(
    f"Mean error reduction  : "
    f"{mean_improvement:.2f}%"
)

print(
    f"Worst error reduction : "
    f"{worst_improvement:.2f}%"
)


# ============================================================
# 6. Pass-rate table
# ============================================================

thresholds = [
    1,
    2,
    5,
    20,
    50
]

pass_rows = []

for threshold in thresholds:

    baseline_pass = np.sum(
        ai["baseline_error"] <= threshold
    )

    ai_pass = np.sum(
        ai["ai_error"] <= threshold
    )

    total = len(ai)

    pass_rows.append(
        {
            "threshold_px": threshold,
            "baseline_pass": baseline_pass,
            "baseline_rate_percent":
                baseline_pass / total * 100,
            "ai_pass": ai_pass,
            "ai_rate_percent":
                ai_pass / total * 100
        }
    )


pass_df = pd.DataFrame(
    pass_rows
)


print()
print("=" * 80)
print("PASS RATE COMPARISON")
print("=" * 80)

print(
    pass_df.to_string(index=False)
)


# ============================================================
# 7. Improvement per sample
# ============================================================

sample_results = (
    ai[
        [
            "sample",
            "noise_level",
            "baseline_error",
            "ai_error"
        ]
    ]
    .copy()
)

sample_results["improvement_px"] = (
    sample_results["baseline_error"]
    -
    sample_results["ai_error"]
)

sample_results["improvement_percent"] = np.where(
    sample_results["baseline_error"] > 0,
    sample_results["improvement_px"]
    /
    sample_results["baseline_error"]
    * 100,
    0
)


# ============================================================
# 8. Best improvements
# ============================================================

print()
print("=" * 80)
print("TOP AI IMPROVEMENTS")
print("=" * 80)

best = (
    sample_results
    .sort_values(
        "improvement_px",
        ascending=False
    )
    .head(10)
)

print(
    best.to_string(index=False)
)


# ============================================================
# 9. Worst remaining AI cases
# ============================================================

print()
print("=" * 80)
print("WORST REMAINING AI CASES")
print("=" * 80)

worst = (
    sample_results
    .sort_values(
        "ai_error",
        ascending=False
    )
    .head(10)
)

print(
    worst.to_string(index=False)
)


# ============================================================
# 10. AI confidence
# ============================================================

if os.path.exists(CONF_FILE):

    confidence = pd.read_csv(
        CONF_FILE
    )

    print()
    print("=" * 80)
    print("AI CONFIDENCE SUMMARY")
    print("=" * 80)

    print(
        confidence[
            [
                "confidence",
                "best_probability",
                "confidence_gap",
                "error"
            ]
        ]
        .groupby("confidence")
        .agg(
            samples=("error", "count"),
            mean_probability=(
                "best_probability",
                "mean"
            ),
            mean_gap=(
                "confidence_gap",
                "mean"
            ),
            mean_error=(
                "error",
                "mean"
            ),
            worst_error=(
                "error",
                "max"
            )
        )
        .to_string()
    )


# ============================================================
# 11. Save outputs
# ============================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

pass_file = os.path.join(
    OUTPUT_DIR,
    "pass_rates.csv"
)

sample_file = os.path.join(
    OUTPUT_DIR,
    "sample_improvements.csv"
)


comparison.to_csv(
    comparison_file
)

pass_df.to_csv(
    pass_file,
    index=False
)

sample_results.to_csv(
    sample_file,
    index=False
)


# ============================================================
# 12. Final summary
# ============================================================

print()
print("=" * 80)
print("DRIFT-SENSE FINAL BENCHMARK COMPLETE")
print("=" * 80)

print()
print(
    f"Baseline mean error : "
    f"{baseline_metrics['mean_error_px']:.3f} px"
)

print(
    f"AI-V2 mean error    : "
    f"{ai_metrics['mean_error_px']:.3f} px"
)

print(
    f"Baseline worst      : "
    f"{baseline_metrics['worst_error_px']:.3f} px"
)

print(
    f"AI-V2 worst         : "
    f"{ai_metrics['worst_error_px']:.3f} px"
)

print()
print(
    f"Mean improvement    : "
    f"{mean_improvement:.2f}%"
)

print(
    f"Worst-case improvement : "
    f"{worst_improvement:.2f}%"
)

print()
print("Saved:")
print(
    f"  {comparison_file}"
)
print(
    f"  {pass_file}"
)
print(
    f"  {sample_file}"
)