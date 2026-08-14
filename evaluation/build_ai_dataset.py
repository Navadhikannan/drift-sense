from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ============================================================
# DRIFT-SENSE AI PHASE
# BUILD CANDIDATE-LEVEL TRAINING DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "ai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "candidate_dataset.csv"
)

TEMPLATE_SIZE = 100

TOP_K = 30

# Candidate considered correct if it is within this
# distance from ground truth.
POSITIVE_RADIUS = 5.0


# ============================================================
# FEATURES
# ============================================================

def make_edges(image):

    return cv2.Canny(
        image,
        50,
        150
    )


def make_gradient(image):

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(
        np.uint8
    )


def correlation(a, b):

    if a.shape != b.shape:
        return 0.0

    a = a.astype(
        np.float32
    )

    b = b.astype(
        np.float32
    )

    a -= np.mean(a)
    b -= np.mean(b)

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator < 1e-8:
        return 0.0

    value = (
        np.sum(a * b)
        /
        denominator
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )


def normalized_correlation(a, b):

    return (
        correlation(a, b)
        + 1.0
    ) / 2.0


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(
    response,
    top_k=TOP_K
):

    candidates = []

    kernel = np.ones(
        (25, 25),
        np.uint8
    )

    local_max = cv2.dilate(
        response,
        kernel
    )

    mask = (
        response
        >= local_max - 1e-6
    )

    ys, xs = np.where(mask)

    points = []

    for y, x in zip(ys, xs):

        points.append(
            (
                float(response[y, x]),
                int(x),
                int(y)
            )
        )

    points.sort(
        key=lambda item: item[0],
        reverse=True
    )

    for score, x, y in points:

        too_close = False

        for _, old_x, old_y in candidates:

            distance = np.sqrt(
                (x - old_x) ** 2
                +
                (y - old_y) ** 2
            )

            if distance < 25:

                too_close = True
                break

        if too_close:
            continue

        candidates.append(
            (
                score,
                x,
                y
            )
        )

        if len(candidates) >= top_k:
            break

    return candidates


# ============================================================
# PROCESS ONE IMAGE PAIR
# ============================================================

def process_pair(
    reference_path,
    search_path,
    sample_name,
    noise_level
):

    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        str(search_path),
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise RuntimeError(
            f"Cannot read {reference_path}"
        )

    if search is None:
        raise RuntimeError(
            f"Cannot read {search_path}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    metadata_path = (
        reference_path.parent
        / "metadata.json"
    )

    if not metadata_path.exists():

        raise RuntimeError(
            f"Missing metadata: "
            f"{metadata_path}"
        )

    metadata = pd.read_json(
        metadata_path,
        typ="series"
    )

    gt_x = float(
        metadata["gt_x"]
    )

    gt_y = float(
        metadata["gt_y"]
    )

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    template = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    template_edges = make_edges(
        template
    )

    template_gradient = make_gradient(
        template
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_edges = make_edges(
        search
    )

    search_gradient = make_gradient(
        search
    )

    response = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    candidates = get_candidates(
        response
    )

    rows = []

    image_center_x = (
        search.shape[1] / 2.0
    )

    image_center_y = (
        search.shape[0] / 2.0
    )

    image_diagonal = np.sqrt(
        search.shape[1] ** 2
        +
        search.shape[0] ** 2
    )

    # --------------------------------------------------------
    # Candidate features
    # --------------------------------------------------------

    for rank, (
        template_score,
        x,
        y
    ) in enumerate(
        candidates,
        start=1
    ):

        patch = search[
            y:y + TEMPLATE_SIZE,
            x:x + TEMPLATE_SIZE
        ]

        patch_edges = search_edges[
            y:y + TEMPLATE_SIZE,
            x:x + TEMPLATE_SIZE
        ]

        patch_gradient = search_gradient[
            y:y + TEMPLATE_SIZE,
            x:x + TEMPLATE_SIZE
        ]

        if (
            patch.shape
            != template.shape
        ):
            continue

        center_x = (
            x + TEMPLATE_SIZE / 2.0
        )

        center_y = (
            y + TEMPLATE_SIZE / 2.0
        )

        edge_score = (
            normalized_correlation(
                template_edges,
                patch_edges
            )
        )

        gradient_score = (
            normalized_correlation(
                template_gradient,
                patch_gradient
            )
        )

        gray_score = (
            normalized_correlation(
                template,
                patch
            )
        )

        # Combined handcrafted score
        structural_score = (
            0.40 * gray_score
            +
            0.20 * edge_score
            +
            0.40 * gradient_score
        )

        distance_to_gt = np.sqrt(
            (center_x - gt_x) ** 2
            +
            (center_y - gt_y) ** 2
        )

        distance_to_center = np.sqrt(
            (center_x - image_center_x) ** 2
            +
            (center_y - image_center_y) ** 2
        )

        normalized_center_distance = (
            distance_to_center
            /
            image_diagonal
        )

        label = int(
            distance_to_gt
            <= POSITIVE_RADIUS
        )

        rows.append(
            {
                "sample": sample_name,
                "noise_level": noise_level,
                "rank": rank,

                "candidate_x": center_x,
                "candidate_y": center_y,

                "gt_x": gt_x,
                "gt_y": gt_y,

                "template_score":
                    template_score,

                "gray_score":
                    gray_score,

                "edge_score":
                    edge_score,

                "gradient_score":
                    gradient_score,

                "structural_score":
                    structural_score,

                "center_distance":
                    normalized_center_distance,

                "distance_to_gt":
                    distance_to_gt,

                "label":
                    label
            }
        )

    return rows


# ============================================================
# FIND ALL DATASETS
# ============================================================

def find_samples():

    samples = []

    # --------------------------------------------------------
    # Original benchmark
    # --------------------------------------------------------

    for i in range(1, 31):

        sample_name = (
            f"sample_{i:03d}"
        )

        sample_dir = (
            DATA_DIR
            / sample_name
        )

        if not sample_dir.exists():
            continue

        samples.append(
            (
                sample_name,
                "clean",
                sample_dir
            )
        )

    # --------------------------------------------------------
    # Robustness datasets
    # --------------------------------------------------------

    robustness_dir = (
        DATA_DIR
        / "robustness"
    )

    for noise_level in [
        "low",
        "medium",
        "high"
    ]:

        level_dir = (
            robustness_dir
            / noise_level
        )

        if not level_dir.exists():
            continue

        for sample_dir in sorted(
            level_dir.iterdir()
        ):

            if not sample_dir.is_dir():
                continue

            samples.append(
                (
                    sample_dir.name,
                    noise_level,
                    sample_dir
                )
            )

    return samples


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "DRIFT-SENSE AI PHASE - "
        "CANDIDATE DATASET"
    )
    print("=" * 75)

    samples = find_samples()

    print()
    print(
        f"Image pairs found : "
        f"{len(samples)}"
    )

    all_rows = []

    for index, (
        sample_name,
        noise_level,
        sample_dir
    ) in enumerate(
        samples,
        start=1
    ):

        reference_path = (
            sample_dir
            / "reference.png"
        )

        search_path = (
            sample_dir
            / "search.png"
        )

        try:

            rows = process_pair(
                reference_path,
                search_path,
                sample_name,
                noise_level
            )

            all_rows.extend(
                rows
            )

            print(
                f"[OK] "
                f"{noise_level:6s} "
                f"{sample_name}: "
                f"{len(rows)} candidates"
            )

        except Exception as exc:

            print(
                f"[ERROR] "
                f"{noise_level:6s} "
                f"{sample_name}: "
                f"{exc}"
            )

    if not all_rows:

        print()
        print(
            "[ERROR] No candidate data generated."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 75)
    print("AI CANDIDATE DATASET COMPLETE")
    print("=" * 75)

    print(
        f"Rows generated : "
        f"{len(df)}"
    )

    print(
        f"Positive rows  : "
        f"{int(df['label'].sum())}"
    )

    print(
        f"Negative rows  : "
        f"{int((df['label'] == 0).sum())}"
    )

    print()
    print(
        "Noise distribution:"
    )

    print(
        df["noise_level"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":

    main()