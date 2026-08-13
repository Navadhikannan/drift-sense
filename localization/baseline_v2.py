import sys
import json
import math
import cv2
import numpy as np


# ============================================================
# 1. Load images
# ============================================================

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


# ============================================================
# 2. Create edge representation
# ============================================================

def edge_image(image):
    # Slight blur reduces sensitivity to sensor noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny detects structural edges
    edges = cv2.Canny(
        blurred,
        threshold1=40,
        threshold2=120
    )

    return edges


# ============================================================
# 3. Resize reference to 10x smaller
# ============================================================

def create_template(reference):

    h, w = reference.shape

    new_w = w // 10
    new_h = h // 10

    template = cv2.resize(
        reference,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return template


# ============================================================
# 4. Find candidate locations
# ============================================================

def find_candidates(search_edges, template_edges, top_n=30):

    result = cv2.matchTemplate(
        search_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED
    )

    candidates = []

    # Make a copy so we can suppress nearby detections
    working = result.copy()

    template_h, template_w = template_edges.shape

    # Suppression radius
    suppression = max(template_h, template_w) // 2

    for _ in range(top_n):

        _, max_score, _, max_location = cv2.minMaxLoc(working)

        x = max_location[0]
        y = max_location[1]

        if max_score < 0.05:
            break

        center_x = x + template_w / 2
        center_y = y + template_h / 2

        candidates.append(
            {
                "x": center_x,
                "y": center_y,
                "score": float(max_score)
            }
        )

        # Suppress area around this candidate
        x1 = max(0, x - suppression)
        y1 = max(0, y - suppression)

        x2 = min(working.shape[1], x + template_w + suppression)
        y2 = min(working.shape[0], y + template_h + suppression)

        working[y1:y2, x1:x2] = -1

    return candidates


# ============================================================
# 5. Select candidate using similarity + center rule
# ============================================================

def select_candidate(candidates, image_shape):

    if not candidates:
        raise RuntimeError("No valid candidates found.")

    height, width = image_shape

    search_center_x = width / 2
    search_center_y = height / 2

    # Best matching score
    best_score = max(c["score"] for c in candidates)

    # Keep candidates reasonably close to the best score.
    #
    # This prevents a very weak candidate near the center
    # from beating a genuinely good candidate.
    score_threshold = max(0.70 * best_score, 0.15)

    good_candidates = [
        c for c in candidates
        if c["score"] >= score_threshold
    ]

    # Among credible candidates, choose the one closest
    # to the center of the search image.
    for candidate in good_candidates:

        dx = candidate["x"] - search_center_x
        dy = candidate["y"] - search_center_y

        candidate["center_distance"] = math.sqrt(
            dx * dx + dy * dy
        )

    selected = min(
        good_candidates,
        key=lambda c: c["center_distance"]
    )

    return selected, good_candidates


# ============================================================
# 6. Load ground truth
# ============================================================

def load_ground_truth(metadata_path):

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    gt_x = float(metadata["gt_x"])
    gt_y = float(metadata["gt_y"])

    return gt_x, gt_y


# ============================================================
# 7. Calculate localization error
# ============================================================

def calculate_error(pred_x, pred_y, gt_x, gt_y):

    return math.sqrt(
        (pred_x - gt_x) ** 2 +
        (pred_y - gt_y) ** 2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 4:

        print(
            "Usage:\n"
            "python baseline_v2.py "
            "<reference.png> <search.png> <metadata.json>"
        )

        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]
    metadata_path = sys.argv[3]

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    reference = load_image(reference_path)
    search = load_image(search_path)

    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    template = create_template(reference)

    # --------------------------------------------------------
    # Edge extraction
    # --------------------------------------------------------

    reference_edges = edge_image(template)
    search_edges = edge_image(search)

    # --------------------------------------------------------
    # Candidate search
    # --------------------------------------------------------

    candidates = find_candidates(
        search_edges,
        reference_edges,
        top_n=30
    )

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    selected, good_candidates = select_candidate(
        candidates,
        search.shape
    )

    pred_x = selected["x"]
    pred_y = selected["y"]
    score = selected["score"]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_x, gt_y = load_ground_truth(metadata_path)

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("=" * 60)
    print("DRIFT-SENSE BASELINE V2")
    print("=" * 60)

    print(f"Reference size      : {reference.shape[1]} x {reference.shape[0]}")
    print(f"Search size         : {search.shape[1]} x {search.shape[0]}")
    print(f"Template size       : {template.shape[1]} x {template.shape[0]}")

    print()
    print(f"Candidates found    : {len(candidates)}")
    print(f"Credible candidates : {len(good_candidates)}")

    print()
    print(f"Predicted center    : ({pred_x:.2f}, {pred_y:.2f})")
    print(f"Ground truth        : ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Localization error  : {error:.2f} pixels")
    print(f"Matching score      : {score:.4f}")

    if "center_distance" in selected:
        print(
            f"Distance from image center: "
            f"{selected['center_distance']:.2f} pixels"
        )

    print("=" * 60)

    # --------------------------------------------------------
    # Print candidate list
    # --------------------------------------------------------

    print("\nTop candidates:")

    for i, candidate in enumerate(
        sorted(
            candidates,
            key=lambda c: c["score"],
            reverse=True
        )[:10],
        start=1
    ):

        distance = math.sqrt(
            (candidate["x"] - search.shape[1] / 2) ** 2 +
            (candidate["y"] - search.shape[0] / 2) ** 2
        )

        print(
            f"{i:2d}. "
            f"({candidate['x']:.1f}, {candidate['y']:.1f}) "
            f"score={candidate['score']:.4f} "
            f"center_distance={distance:.1f}"
        )


if __name__ == "__main__":
    main()