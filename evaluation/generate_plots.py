import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "results/evaluation_results.csv"
OUTPUT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DRIFT-SENSE FINAL RESULT VISUALIZATION")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

print(f"Samples loaded : {len(df)}")
print(f"Columns        : {', '.join(df.columns)}")
print(f"Output folder  : {OUTPUT_DIR}")
print()


# =========================================================
# 1. V1 PER-SAMPLE ERROR
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    range(1, len(df) + 1),
    df["v1_error"],
    marker="o"
)

plt.axhline(
    5,
    linestyle="--",
    label="5 px threshold"
)

plt.xlabel("Sample")
plt.ylabel("Localization Error (pixels)")
plt.title(
    "DRIFT-SENSE - V1 Per-Sample Localization Error"
)

plt.xticks(
    range(1, len(df) + 1)
)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "v1_per_sample_error.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"[OK] {path}")


# =========================================================
# 2. V1 PASS RATE
# =========================================================

thresholds = [1, 2, 4, 5]

pass_rates = []

for threshold in thresholds:

    passed = (
        df["v1_error"] <= threshold
    ).sum()

    rate = (
        passed / len(df)
    ) * 100

    pass_rates.append(rate)


plt.figure(figsize=(9, 6))

bars = plt.bar(
    [f"{x} px" for x in thresholds],
    pass_rates
)

plt.xlabel("Localization Error Tolerance")
plt.ylabel("Pass Rate (%)")

plt.title(
    "DRIFT-SENSE - V1 Localization Pass Rate"
)

plt.ylim(0, 100)

for bar, value in zip(
    bars,
    pass_rates
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "v1_pass_rate.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"[OK] {path}")


# =========================================================
# 3. V1 ERROR DISTRIBUTION
# =========================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["v1_error"],
    bins=15
)

plt.xlabel("Localization Error (pixels)")
plt.ylabel("Number of Samples")

plt.title(
    "DRIFT-SENSE - V1 Error Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "v1_error_distribution.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"[OK] {path}")


# =========================================================
# 4. V1 vs V3 ERROR
# =========================================================

plt.figure(figsize=(12, 6))

x = range(1, len(df) + 1)

plt.plot(
    x,
    df["v1_error"],
    marker="o",
    label="V1 - Template Matching"
)

plt.plot(
    x,
    df["v3_error"],
    marker="x",
    label="V3 - SIFT"
)

plt.xlabel("Sample")
plt.ylabel("Localization Error (pixels)")

plt.title(
    "DRIFT-SENSE - V1 vs V3 Localization Error"
)

plt.xticks(x)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "v1_vs_v3_error.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"[OK] {path}")


# =========================================================
# 5. RUNTIME COMPARISON
# =========================================================

v1_runtime_ms = (
    df["v1_runtime_sec"].mean()
    * 1000
)

v3_runtime_ms = (
    df["v3_runtime_sec"].mean()
    * 1000
)

methods = [
    "V1\nTemplate Matching",
    "V3\nSIFT"
]

runtime_values = [
    v1_runtime_ms,
    v3_runtime_ms
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    methods,
    runtime_values
)

plt.ylabel("Mean Runtime (ms)")

plt.title(
    "DRIFT-SENSE - Runtime Comparison"
)

for bar, value in zip(
    bars,
    runtime_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 5,
        f"{value:.1f} ms",
        ha="center"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "runtime_comparison_v1_v3.png"
)

plt.savefig(
    path,
    dpi=300
)

plt.close()

print(f"[OK] {path}")


# =========================================================
# 6. FINAL STATISTICS
# =========================================================

mean_error = df["v1_error"].mean()
median_error = df["v1_error"].median()
worst_error = df["v1_error"].max()

print()
print("=" * 70)
print("FINAL V1 STATISTICS")
print("=" * 70)

print(
    f"Mean error       : {mean_error:.3f} px"
)

print(
    f"Median error     : {median_error:.3f} px"
)

print(
    f"Worst-case error : {worst_error:.3f} px"
)

print(
    f"Mean runtime     : {v1_runtime_ms:.3f} ms"
)

print()

for threshold in thresholds:

    passed = (
        df["v1_error"] <= threshold
    ).sum()

    rate = (
        passed / len(df)
    ) * 100

    print(
        f"Pass rate @ {threshold}px : "
        f"{passed}/{len(df)} "
        f"({rate:.2f}%)"
    )


# =========================================================
# 7. HARDEST SAMPLES
# =========================================================

print()
print("=" * 70)
print("HARDEST V1 SAMPLES")
print("=" * 70)

hardest = df.sort_values(
    "v1_error",
    ascending=False
)

for _, row in hardest.head(5).iterrows():

    print(
        f"{row['sample']}: "
        f"{row['v1_error']:.2f} px"
    )


print()
print("=" * 70)
print("PLOTS GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    f"Location: {OUTPUT_DIR}"
)