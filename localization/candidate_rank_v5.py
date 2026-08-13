import sys
import cv2
import numpy as np


TEMPLATE_SIZE = 100
TOP_K = 30

# Weights for the three representations
GRAY_WEIGHT = 0.50
EDGE_WEIGHT = 0.25
GRADIENT_WEIGHT = 0.25


def normalize_scores(scores):
    """
    Normalize scores to 0..1 so that different
    representations can be combined.
    """
    scores = np.asarray(scores, dtype=np.float32)

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum - minimum < 1e-8:
        return np.ones_like(scores)

    return (scores - minimum) / (maximum - minimum)


def create_edges(image):
    """
    Create Canny edge representation.
    """
    return cv2.Canny(
        image,
        50,
        150
    )


def create_gradient(image):
    """
    Create gradient magnitude representation.
    """
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

    magnitude = cv2.magnitude(gx, gy)

    # Normalize to 8-bit
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)


def get_candidates(response, top_k=TOP_K):
    """
    Extract spatially separated local maxima
    from the template-matching response.
    """

    candidates = []

    # Local maximum filter.
    kernel_size = 25

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    local_max = cv2.dilate(
        response,
        kernel
    )

    # Pixels that are local maxima
    mask = (
        response >= local_max - 1e-6
    )

    ys, xs = np.where(mask)

    candidate_pixels = []

    for x, y in zip(xs, ys):

        candidate_pixels.append(
            (
                float(response[y, x]),
                int(x),
                int(y)
            )
        )

    # Highest scores first
    candidate_pixels.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Non-maximum suppression
    min_distance = 40

    for score, x, y in candidate_pixels:

        too_close = False

        for _, selected_x, selected_y in candidates:

            distance = np.sqrt(
                (x - selected_x) ** 2 +
                (y - selected_y) ** 2
            )

            if distance < min_distance:
                too_close = True
                break

        if not too_close:

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


def score_candidate(
    gray_image,
    edge_image,
    gradient_image,
    gray_template,
    edge_template,
    gradient_template,
    x,
    y
):

    h, w = gray_template.shape

    gray_patch = gray_image[
        y:y + h,
        x:x + w
    ]

    edge_patch = edge_image[
        y:y + h,
        x:x + w
    ]

    gradient_patch = gradient_image[
        y:y + h,
        x:x + w
    ]

    # Safety check
    if (
        gray_patch.shape != gray_template.shape
        or edge_patch.shape != edge_template.shape
        or gradient_patch.shape != gradient_template.shape
    ):
        return None

    gray_result = cv2.matchTemplate(
        gray_patch,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    edge_result = cv2.matchTemplate(
        edge_patch,
        edge_template,
        cv2.TM_CCOEFF_NORMED
    )

    gradient_result = cv2.matchTemplate(
        gradient_patch,
        gradient_template,
        cv2.TM_CCOEFF_NORMED
    )

    gray_score = float(gray_result[0, 0])
    edge_score = float(edge_result[0, 0])
    gradient_score = float(
        gradient_result[0, 0]
    )

    return (
        gray_score,
        edge_score,
        gradient_score
    )


def run_v5(reference_path, search_path):

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
            f"Could not read reference: {reference_path}"
        )

    if search is None:
        raise FileNotFoundError(
            f"Could not read search: {search_path}"
        )

    # ---------------------------------------------------------
    # Resize reference to the same 100x100 template
    # used by V1.
    # ---------------------------------------------------------

    gray_template = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------------------------
    # Create representations
    # ---------------------------------------------------------

    edge_template = create_edges(
        gray_template
    )

    gradient_template = create_gradient(
        gray_template
    )

    edge_search = create_edges(
        search
    )

    gradient_search = create_gradient(
        search
    )

    # ---------------------------------------------------------
    # Stage 1:
    # Generate candidates using normal grayscale
    # template matching.
    # ---------------------------------------------------------

    gray_response = cv2.matchTemplate(
        search,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    candidates = get_candidates(
        gray_response,
        TOP_K
    )

    if not candidates:

        raise RuntimeError(
            "No candidates found."
        )

    # ---------------------------------------------------------
    # Stage 2:
    # Calculate additional scores for every candidate.
    # ---------------------------------------------------------

    raw_results = []

    for gray_score, x, y in candidates:

        scores = score_candidate(
            search,
            edge_search,
            gradient_search,
            gray_template,
            edge_template,
            gradient_template,
            x,
            y
        )

        if scores is None:
            continue

        gray_local, edge_local, gradient_local = scores

        raw_results.append(
            {
                "x": x,
                "y": y,
                "gray": gray_local,
                "edge": edge_local,
                "gradient": gradient_local
            }
        )

    if not raw_results:

        raise RuntimeError(
            "Could not score candidates."
        )

    # ---------------------------------------------------------
    # Normalize each representation independently.
    # ---------------------------------------------------------

    gray_scores = normalize_scores(
        [r["gray"] for r in raw_results]
    )

    edge_scores = normalize_scores(
        [r["edge"] for r in raw_results]
    )

    gradient_scores = normalize_scores(
        [r["gradient"] for r in raw_results]
    )

    # ---------------------------------------------------------
    # Combined score
    # ---------------------------------------------------------

    for i, result in enumerate(raw_results):

        result["gray_norm"] = float(
            gray_scores[i]
        )

        result["edge_norm"] = float(
            edge_scores[i]
        )

        result["gradient_norm"] = float(
            gradient_scores[i]
        )

        result["combined"] = (
            GRAY_WEIGHT *
            result["gray_norm"]
            +
            EDGE_WEIGHT *
            result["edge_norm"]
            +
            GRADIENT_WEIGHT *
            result["gradient_norm"]
        )

        # Convert top-left to center
        result["center_x"] = (
            result["x"]
            + TEMPLATE_SIZE / 2.0
        )

        result["center_y"] = (
            result["y"]
            + TEMPLATE_SIZE / 2.0
        )

    # ---------------------------------------------------------
    # Sort by combined score
    # ---------------------------------------------------------

    raw_results.sort(
        key=lambda r: r["combined"],
        reverse=True
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("DRIFT-SENSE V5 - CANDIDATE RANKING")
    print("=" * 100)

    print(
        f"Reference size : "
        f"{reference.shape[1]} x {reference.shape[0]}"
    )

    print(
        f"Search size    : "
        f"{search.shape[1]} x {search.shape[0]}"
    )

    print(
        f"Template size  : "
        f"{TEMPLATE_SIZE} x {TEMPLATE_SIZE}"
    )

    print(
        f"Candidates     : "
        f"{len(raw_results)}"
    )

    print()

    print(
        f"{'Rank':<6}"
        f"{'X':<10}"
        f"{'Y':<10}"
        f"{'Gray':<10}"
        f"{'Edge':<10}"
        f"{'Grad':<10}"
        f"{'Combined':<12}"
    )

    print("-" * 100)

    for rank, result in enumerate(
        raw_results,
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{result['center_x']:<10.2f}"
            f"{result['center_y']:<10.2f}"
            f"{result['gray']:<10.4f}"
            f"{result['edge']:<10.4f}"
            f"{result['gradient']:<10.4f}"
            f"{result['combined']:<12.4f}"
        )

    # ---------------------------------------------------------
    # Final V5 prediction
    # ---------------------------------------------------------

    best = raw_results[0]

    print()
    print("=" * 100)
    print("V5 PREDICTION")
    print("=" * 100)

    print(
        f"Predicted center : "
        f"({best['center_x']:.2f}, "
        f"{best['center_y']:.2f})"
    )

    print(
        f"Combined score   : "
        f"{best['combined']:.4f}"
    )

    print(
        f"Gray score       : "
        f"{best['gray']:.4f}"
    )

    print(
        f"Edge score       : "
        f"{best['edge']:.4f}"
    )

    print(
        f"Gradient score   : "
        f"{best['gradient']:.4f}"
    )

    print("=" * 100)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python localization\\candidate_rank_v5.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    run_v5(
        sys.argv[1],
        sys.argv[2]
    )