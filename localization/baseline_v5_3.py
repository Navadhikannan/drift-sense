import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DRIFT-SENSE V5.3
# MULTI-SCALE CONTEXTUAL VERIFICATION
# ============================================================

TEMPLATE_SIZE = 100

# Candidate count
TOP_K = 30

# Context windows
LOCAL_SIZE = 100
CONTEXT_SIZE = 300
REGIONAL_SIZE = 500

# Score weights
LOCAL_WEIGHT = 0.50
CONTEXT_WEIGHT = 0.30
REGIONAL_WEIGHT = 0.20


# ============================================================
# CORRELATION
# ============================================================

def correlation(a, b):

    if a is None or b is None:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a -= np.mean(a)
    b -= np.mean(b)

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator < 1e-8:
        return 0.0

    value = np.sum(a * b) / denominator

    return float(
        np.clip(value, -1.0, 1.0)
    )


def normalized_correlation(a, b):

    value = correlation(a, b)

    return (value + 1.0) / 2.0


# ============================================================
# IMAGE FEATURES
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


# ============================================================
# CENTERED PATCH EXTRACTION
# ============================================================

def extract_center_patch(
    image,
    center_x,
    center_y,
    size
):

    half = size // 2

    x1 = int(
        round(center_x) - half
    )

    y1 = int(
        round(center_y) - half
    )

    x2 = x1 + size
    y2 = y1 + size

    if (
        x1 < 0
        or y1 < 0
        or x2 > image.shape[1]
        or y2 > image.shape[0]
    ):
        return None

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# PATCH SIMILARITY
# ============================================================

def patch_similarity(
    reference_patch,
    search_patch
):

    if (
        reference_patch is None
        or search_patch is None
    ):
        return 0.0

    if (
        reference_patch.shape
        != search_patch.shape
    ):
        return 0.0

    # Gray
    gray_score = normalized_correlation(
        reference_patch,
        search_patch
    )

    # Edge
    reference_edges = make_edges(
        reference_patch
    )

    search_edges = make_edges(
        search_patch
    )

    edge_score = normalized_correlation(
        reference_edges,
        search_edges
    )

    # Gradient
    reference_gradient = make_gradient(
        reference_patch
    )

    search_gradient = make_gradient(
        search_patch
    )

    gradient_score = normalized_correlation(
        reference_gradient,
        search_gradient
    )

    # Combined
    score = (
        0.40 * gray_score
        +
        0.20 * edge_score
        +
        0.40 * gradient_score
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ============================================================
# TOP CANDIDATES
# ============================================================

def get_top_candidates(
    response,
    top_k=TOP_K
):

    candidates = []

    kernel_size = 25

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        np.uint8
    )

    local_max = cv2.dilate(
        response,
        kernel
    )

    mask = (
        response >=
        local_max - 1e-6
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
# BUILD REFERENCE CONTEXTS
# ============================================================

def build_reference_contexts(
    reference
):

    center_x = (
        reference.shape[1] / 2.0
    )

    center_y = (
        reference.shape[0] / 2.0
    )

    local = extract_center_patch(
        reference,
        center_x,
        center_y,
        LOCAL_SIZE
    )

    context = extract_center_patch(
        reference,
        center_x,
        center_y,
        CONTEXT_SIZE
    )

    regional = extract_center_patch(
        reference,
        center_x,
        center_y,
        REGIONAL_SIZE
    )

    return (
        local,
        context,
        regional
    )


# ============================================================
# EVALUATE CANDIDATE
# ============================================================

def evaluate_candidate(
    reference,
    search,
    center_x,
    center_y,
    template_score
):

    reference_local, \
    reference_context, \
    reference_regional = (
        build_reference_contexts(
            reference
        )
    )

    search_local = extract_center_patch(
        search,
        center_x,
        center_y,
        LOCAL_SIZE
    )

    search_context = extract_center_patch(
        search,
        center_x,
        center_y,
        CONTEXT_SIZE
    )

    search_regional = extract_center_patch(
        search,
        center_x,
        center_y,
        REGIONAL_SIZE
    )

    if (
        search_local is None
        or search_context is None
        or search_regional is None
    ):
        return None

    # --------------------------------------------------------
    # Three scales
    # --------------------------------------------------------

    local_score = patch_similarity(
        reference_local,
        search_local
    )

    context_score = patch_similarity(
        reference_context,
        search_context
    )

    regional_score = patch_similarity(
        reference_regional,
        search_regional
    )

    # --------------------------------------------------------
    # Multi-scale score
    # --------------------------------------------------------

    multiscale_score = (
        LOCAL_WEIGHT * local_score
        +
        CONTEXT_WEIGHT * context_score
        +
        REGIONAL_WEIGHT * regional_score
    )

    # --------------------------------------------------------
    # Combine original template score with
    # multi-scale structural score
    # --------------------------------------------------------

    final_score = (
        0.50 * template_score
        +
        0.50 * multiscale_score
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "template_score": template_score,
        "local_score": local_score,
        "context_score": context_score,
        "regional_score": regional_score,
        "multiscale_score": multiscale_score,
        "final_score": final_score
    }


# ============================================================
# MAIN V5.3
# ============================================================

def run_v5_3(
    reference_path,
    search_path
):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:

        raise FileNotFoundError(
            f"Could not read reference image: "
            f"{reference_path}"
        )

    if search is None:

        raise FileNotFoundError(
            f"Could not read search image: "
            f"{search_path}"
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

    # --------------------------------------------------------
    # Template matching
    # --------------------------------------------------------

    response = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    candidates = get_top_candidates(
        response,
        TOP_K
    )

    if len(candidates) == 0:

        raise RuntimeError(
            "No candidates found."
        )

    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    evaluated = []

    for rank, (
        template_score,
        x,
        y
    ) in enumerate(
        candidates,
        start=1
    ):

        center_x = (
            x + TEMPLATE_SIZE / 2.0
        )

        center_y = (
            y + TEMPLATE_SIZE / 2.0
        )

        result = evaluate_candidate(
            reference,
            search,
            center_x,
            center_y,
            template_score
        )

        if result is None:
            continue

        result["rank"] = rank

        evaluated.append(
            result
        )

    if not evaluated:

        raise RuntimeError(
            "No valid candidates after "
            "multi-scale verification."
        )

    # --------------------------------------------------------
    # Winners
    # --------------------------------------------------------

    original_best = max(
        evaluated,
        key=lambda item:
        item["template_score"]
    )

    best = max(
        evaluated,
        key=lambda item:
        item["final_score"]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "DRIFT-SENSE V5.3 - "
        "MULTI-SCALE CONTEXTUAL VERIFICATION"
    )
    print("=" * 100)

    print(
        f"Reference size : "
        f"{reference.shape[1]} x "
        f"{reference.shape[0]}"
    )

    print(
        f"Search size    : "
        f"{search.shape[1]} x "
        f"{search.shape[0]}"
    )

    print(
        f"Template size  : "
        f"{TEMPLATE_SIZE} x "
        f"{TEMPLATE_SIZE}"
    )

    print(
        f"Candidates     : "
        f"{len(evaluated)}"
    )

    print()

    print(
        "Rank  X       Y       "
        "Template  Local   Context  Regional  Final"
    )

    print("-" * 100)

    ranked = sorted(
        evaluated,
        key=lambda item:
        item["final_score"],
        reverse=True
    )

    for display_rank, item in enumerate(
        ranked[:15],
        start=1
    ):

        print(
            f"{display_rank:<6}"
            f"{item['center_x']:<8.2f}"
            f"{item['center_y']:<8.2f}"
            f"{item['template_score']:<10.4f}"
            f"{item['local_score']:<8.4f}"
            f"{item['context_score']:<9.4f}"
            f"{item['regional_score']:<10.4f}"
            f"{item['final_score']:.4f}"
        )

    print()

    print("=" * 100)
    print("ORIGINAL TEMPLATE-MATCH WINNER")
    print("-" * 100)

    print(
        f"Center          : "
        f"({original_best['center_x']:.2f}, "
        f"{original_best['center_y']:.2f})"
    )

    print(
        f"Template score  : "
        f"{original_best['template_score']:.4f}"
    )

    print()

    print("V5.3 WINNER")
    print("-" * 100)

    print(
        f"Center          : "
        f"({best['center_x']:.2f}, "
        f"{best['center_y']:.2f})"
    )

    print(
        f"Template score  : "
        f"{best['template_score']:.4f}"
    )

    print(
        f"Local score     : "
        f"{best['local_score']:.4f}"
    )

    print(
        f"Context score   : "
        f"{best['context_score']:.4f}"
    )

    print(
        f"Regional score  : "
        f"{best['regional_score']:.4f}"
    )

    print(
        f"Multi-scale     : "
        f"{best['multiscale_score']:.4f}"
    )

    print(
        f"Final score     : "
        f"{best['final_score']:.4f}"
    )

    print("=" * 100)

    return {
        "x": best["center_x"],
        "y": best["center_y"],
        "score": best["final_score"],
        "template_score":
            best["template_score"],
        "local_score":
            best["local_score"],
        "context_score":
            best["context_score"],
        "regional_score":
            best["regional_score"],
        "multiscale_score":
            best["multiscale_score"],
        "candidates":
            evaluated
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python "
            "localization\\baseline_v5_3.py "
            "<reference.png> "
            "<search.png>"
        )

        sys.exit(1)

    result = run_v5_3(
        sys.argv[1],
        sys.argv[2]
    )

    print()

    print(
        f"Final predicted center : "
        f"({result['x']:.2f}, "
        f"{result['y']:.2f})"
    )