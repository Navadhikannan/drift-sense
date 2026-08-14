from pathlib import Path
import cv2
import numpy as np
import shutil

# ============================================================
# DRIFT-SENSE PHASE 6
# Robustness Dataset Generator
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data" / "robustness"

SAMPLES = 30

# Noise levels
NOISE_LEVELS = {
    "clean": 0.0,
    "low": 3.0,
    "medium": 7.0,
    "high": 12.0,
}


def add_gaussian_noise(image, sigma, rng):
    """Add reproducible Gaussian detector noise."""

    if sigma == 0:
        return image.copy()

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():

    print("=" * 70)
    print("DRIFT-SENSE PHASE 6 - ROBUSTNESS DATASET")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    for sample_id in range(1, SAMPLES + 1):

        sample_name = f"sample_{sample_id:03d}"
        source = SOURCE_DIR / sample_name

        reference_path = source / "reference.png"
        search_path = source / "search.png"
        metadata_path = source / "metadata.json"

        if not reference_path.exists() or not search_path.exists():
            print(f"[SKIP] {sample_name} - missing image")
            continue

        reference = cv2.imread(
            str(reference_path),
            cv2.IMREAD_UNCHANGED
        )

        search = cv2.imread(
            str(search_path),
            cv2.IMREAD_UNCHANGED
        )

        if reference is None or search is None:
            print(f"[SKIP] {sample_name} - image read failure")
            continue

        # Keep the original reference unchanged.
        for level, sigma in NOISE_LEVELS.items():

            output_sample = (
                OUTPUT_DIR /
                level /
                sample_name
            )

            output_sample.mkdir(
                parents=True,
                exist_ok=True
            )

            # Reproducible seed per sample/condition.
            seed = 600000 + sample_id * 100 + int(sigma * 10)

            rng = np.random.default_rng(seed)

            robust_search = add_gaussian_noise(
                search,
                sigma,
                rng
            )

            cv2.imwrite(
                str(output_sample / "reference.png"),
                reference
            )

            cv2.imwrite(
                str(output_sample / "search.png"),
                robust_search
            )

            # Preserve the original metadata.
            if metadata_path.exists():
                shutil.copy2(
                    metadata_path,
                    output_sample / "metadata.json"
                )

            # Store robustness information separately.
            with open(
                output_sample / "robustness.txt",
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"condition=noise\n"
                    f"level={level}\n"
                    f"gaussian_sigma={sigma}\n"
                    f"seed={seed}\n"
                    f"source={sample_name}\n"
                )

            total += 1

        print(f"[OK] {sample_name}")

    print()
    print("=" * 70)
    print("ROBUSTNESS DATASET COMPLETE")
    print("=" * 70)
    print(f"Generated pairs : {total}")
    print(f"Output folder   : {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()