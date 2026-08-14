import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DRIFT-SENSE - FINAL VISUALIZATION PACKAGE
# ============================================================

OUTPUT_DIR = "results/final_benchmark/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

MODEL_FILE = "results/final_benchmark/model_comparison.csv"
PASS_FILE = "results/final_benchmark/pass_rates.csv"
IMPROVEMENT_FILE = "results/final_benchmark/sample_improvements.csv"

model = pd.read_csv(MODEL_FILE)
passes = pd.read_csv(PASS_FILE)
improvements = pd.read_csv(IMPROVEMENT_FILE)


print("=" * 80)
print("DRIFT-SENSE FINAL VISUALIZATION")
print("=" * 80)

print()
print("Model comparison:")
print(model)

print()
print("Pass rates:")
print(passes)

print()
print("Improvement rows:", len(improvements))


# ============================================================
# 1. MEAN ERROR COMPARISON
# ============================================================

if "metric" in model.columns:

    mean_row = model[
        model["metric"].astype(str).str.contains(
            "mean_error",
            case=False,
            na=False
        )
    ]

    if len(mean_row) > 0:

        baseline = float(
            mean_row["Baseline_V5.1"].iloc[0]
        )

        ai = float(
            mean_row["AI_V2"].iloc[0]
        )

    else:
        baseline = 13.691222
        ai = 3.982513

else:

    baseline = 13.691222
    ai = 3.982513


plt.figure(figsize=(8, 6))

plt.bar(
    ["V5.1 Baseline", "AI-V2"],
    [baseline, ai]
)

plt.ylabel("Mean Localization Error (px)")
plt.title("Mean Localization Error")

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "01_mean_error_comparison.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("[OK]", path)


# ============================================================
# 2. WORST-CASE ERROR
# ============================================================

if "metric" in model.columns:

    worst_row = model[
        model["metric"].astype(str).str.contains(
            "worst_error",
            case=False,
            na=False
        )
    ]

    if len(worst_row) > 0:

        baseline_worst = float(
            worst_row["Baseline_V5.1"].iloc[0]
        )

        ai_worst = float(
            worst_row["AI_V2"].iloc[0]
        )

    else:

        baseline_worst = 629.048313
        ai_worst = 71.731862

else:

    baseline_worst = 629.048313
    ai_worst = 71.731862


plt.figure(figsize=(8, 6))

plt.bar(
    ["V5.1 Baseline", "AI-V2"],
    [baseline_worst, ai_worst]
)

plt.ylabel("Worst Localization Error (px)")
plt.title("Worst-Case Localization Error")

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "02_worst_case_error.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("[OK]", path)


# ============================================================
# 3. PASS RATE COMPARISON
# ============================================================

required_columns = [
    "threshold_px",
    "baseline_rate_percent",
    "ai_rate_percent"
]

if all(column in passes.columns for column in required_columns):

    plt.figure(figsize=(9, 6))

    plt.plot(
        passes["threshold_px"],
        passes["baseline_rate_percent"],
        marker="o",
        label="V5.1 Baseline"
    )

    plt.plot(
        passes["threshold_px"],
        passes["ai_rate_percent"],
        marker="o",
        label="AI-V2"
    )

    plt.xlabel("Error Threshold (px)")
    plt.ylabel("Pass Rate (%)")

    plt.title("Localization Pass Rate Comparison")

    plt.xticks(
        passes["threshold_px"]
    )

    plt.ylim(0, 105)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "03_pass_rate_comparison.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("[OK]", path)


# ============================================================
# 4. PER-SAMPLE IMPROVEMENT
# ============================================================

if "improvement_px" in improvements.columns:

    grouped = (
        improvements
        .groupby("sample", as_index=False)
        ["improvement_px"]
        .mean()
        .sort_values(
            "improvement_px",
            ascending=False
        )
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        grouped["sample"],
        grouped["improvement_px"]
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel("Sample")
    plt.ylabel("Error Reduction (px)")

    plt.title(
        "AI-V2 Improvement Over V5.1"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "04_per_sample_improvement.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("[OK]", path)


# ============================================================
# 5. TOP IMPROVEMENTS
# ============================================================

if "improvement_px" in improvements.columns:

    top = (
        improvements
        .sort_values(
            "improvement_px",
            ascending=False
        )
        .head(15)
    )

    plt.figure(figsize=(10, 7))

    labels = (
        top["sample"].astype(str)
        + " - "
        + top["noise_level"].astype(str)
    )

    plt.barh(
        labels,
        top["improvement_px"]
    )

    plt.xlabel("Error Reduction (px)")

    plt.title(
        "Top 15 AI-V2 Improvements"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "05_top_improvements.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print("[OK]", path)


# ============================================================
# FINAL SUMMARY
# ============================================================

mean_improvement = (
    (baseline - ai)
    / baseline
    * 100
)

worst_improvement = (
    (baseline_worst - ai_worst)
    / baseline_worst
    * 100
)


print()
print("=" * 80)
print("FINAL VISUALIZATION COMPLETE")
print("=" * 80)

print()
print(
    f"Baseline mean error : {baseline:.3f} px"
)

print(
    f"AI-V2 mean error    : {ai:.3f} px"
)

print(
    f"Mean improvement    : {mean_improvement:.2f}%"
)

print()

print(
    f"Baseline worst      : {baseline_worst:.3f} px"
)

print(
    f"AI-V2 worst         : {ai_worst:.3f} px"
)

print(
    f"Worst improvement   : {worst_improvement:.2f}%"
)

print()
print("Plots saved to:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print()