import sys
import cv2
import numpy as np


TEMPLATE_SIZE = 100

# Number of candidates considered for verification
TOP_K = 10

# If the difference between the first and second
# candidate is larger than this, V1 is considered confident.
CONFIDENCE_GAP = 0.015

# Alternative candidate must beat the original by
# at least this amount in structural score.
MIN_STRUCTURAL_IMPROVEMENT = 0.03


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

    return magnitude.astype(np.uint8)


def get_top_candidates(response, top_k=TOP_K):
    """
    Get spatially separated local maxima.
    """

    candidates = []

    kernel_size = 25

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    local_max = cv2.dilate(
        response,
        kernel
    )

    mask = response >= local_max - 1e-6

    ys, xs = np.where(mask)

    possible = []

    for x, y in zip(xs, ys):

        possible.append(
            (
                float(response[y, x]),
                int(x),
                int(y)
            )
        )

    possible.sort(
        key=lambda item: item[0],
        reverse=True
    )

    min_distance = 40

    for score, x, y in possible:

        close = False

        for _, old_x, old_y in candidates:

            distance = np.sqrt(
                (x - old_x) ** 2 +
                (y - old_y) ** 2
            )

            if distance < min_distance:
                close = True
                break

        if not close:

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


def correlation(template, patch):
    """
    Calculate normalized correlation safely.
    """

    if patch.shape != template.shape:
        return -1.0

    result = cv2.matchTemplate(
        patch,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    return float(result[0, 0])


def evaluate_candidate(
    search_gray,
    search_edges,
    search_gradient,
    template_gray,
    template_edges,
    template_gradient,
    x,
    y
):

    h, w = template_gray.shape

    gray_patch = search_gray[
        y:y + h,
        x:x + w
    ]

    edge_patch = search_edges[
        y:y + h,
        x:x + w
    ]

    gradient_patch = search_gradient[
        y:y + h,
        x:x + w
    ]

    if (
        gray_patch.shape != template_gray.shape
        or edge_patch.shape != template_edges.shape
        or gradient_patch.shape != template_gradient.shape
    ):
        return None

    gray_score = correlation(
        template_gray,
        gray_patch
    )

    edge_score = correlation(
        template_edges,
        edge_patch
    )

    gradient_score = correlation(
        template_gradient,
        gradient_patch
    )

    # Structural score deliberately does NOT use
    # min-max normalization.
    #
    # This avoids the problem encountered in V5,
    # where normalization exaggerated small
    # differences between candidates.

    structural_score = (
        0.50 * gray_score
        + 0.25 * edge_score
        + 0.25 * gradient_score
    )

    return {
        "gray": gray_score,
        "edge": edge_score,
        "gradient": gradient_score,
        "structural": structural_score
    }


def run_v5_1(reference_path, search_path):

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

    # ---------------------------------------------------------
    # 1. Resize reference
    # ---------------------------------------------------------

    template_gray = cv2.resize(
        reference,
        (
            TEMPLATE_SIZE,
            TEMPLATE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------------------------
    # 2. Generate image representations
    # ---------------------------------------------------------

    template_edges = make_edges(
        template_gray
    )

    template_gradient = make_gradient(
        template_gray
    )

    search_edges = make_edges(
        search
    )

    search_gradient = make_gradient(
        search
    )

    # ---------------------------------------------------------
    # 3. Original V1 template matching
    # ---------------------------------------------------------

    response = cv2.matchTemplate(
        search,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, v1_score, _, v1_location = cv2.minMaxLoc(
        response
    )

    v1_x = (
        v1_location[0]
        + TEMPLATE_SIZE / 2.0
    )

    v1_y = (
        v1_location[1]
        + TEMPLATE_SIZE / 2.0
    )

    # ---------------------------------------------------------
    # 4. Generate candidate list
    # ---------------------------------------------------------

    candidates = get_top_candidates(
        response,
        TOP_K
    )

    if len(candidates) == 0:

        raise RuntimeError(
            "No candidates found."
        )

    # ---------------------------------------------------------
    # 5. Determine confidence of V1
    # ---------------------------------------------------------

    first_score = candidates[0][0]

    second_score = (
        candidates[1][0]
        if len(candidates) > 1
        else -1.0
    )

    score_gap = (
        first_score - second_score
    )

    # ---------------------------------------------------------
    # 6. Evaluate candidates structurally
    # ---------------------------------------------------------

    evaluated = []

    for rank, (
        gray_score,
        x,
        y
    ) in enumerate(
        candidates,
        start=1
    ):

        scores = evaluate_candidate(
            search,
            search_edges,
            search_gradient,
            template_gray,
            template_edges,
            template_gradient,
            x,
            y
        )

        if scores is None:
            continue

        center_x = (
            x + TEMPLATE_SIZE / 2.0
        )

        center_y = (
            y + TEMPLATE_SIZE / 2.0
        )

        evaluated.append(
            {
                "rank": rank,
                "x": x,
                "y": y,
                "center_x": center_x,
                "center_y": center_y,
                "v1_score": gray_score,
                "edge": scores["edge"],
                "gradient": scores["gradient"],
                "structural": scores["structural"]
            }
        )

    if not evaluated:

        raise RuntimeError(
            "Candidate verification failed."
        )

    # ---------------------------------------------------------
    # 7. Find best structural candidate
    # ---------------------------------------------------------

    structural_best = max(
        evaluated,
        key=lambda item: item["structural"]
    )

    # Structural score of original V1 winner
    v1_candidate = min(
        evaluated,
        key=lambda item: np.sqrt(
            (item["center_x"] - v1_x) ** 2
            +
            (item["center_y"] - v1_y) ** 2
        )
    )

    structural_improvement = (
        structural_best["structural"]
        - v1_candidate["structural"]
    )

    # ---------------------------------------------------------
    # 8. Conservative decision
    # ---------------------------------------------------------

    use_v1 = True
    decision_reason = ""

    if score_gap >= CONFIDENCE_GAP:

        # V1 has a clear winner.
        use_v1 = True

        decision_reason = (
            "V1 winner is sufficiently confident."
        )

    else:

        # Ambiguous V1 result.
        #
        # Only switch if structural evidence
        # is meaningfully better.

        if (
            structural_best["rank"]
            != v1_candidate["rank"]
            and
            structural_improvement
            >= MIN_STRUCTURAL_IMPROVEMENT
        ):

            use_v1 = False

            decision_reason = (
                "Ambiguous V1 result; "
                "structural verification improved "
                "the candidate."
            )

        else:

            use_v1 = True

            decision_reason = (
                "Ambiguous result, but structural "
                "evidence is not strong enough to "
                "override V1."
            )

    # ---------------------------------------------------------
    # 9. Final prediction
    # ---------------------------------------------------------

    if use_v1:

        final_x = v1_x
        final_y = v1_y

    else:

        final_x = structural_best["center_x"]
        final_y = structural_best["center_y"]

    # ---------------------------------------------------------
    # 10. Print results
    # ---------------------------------------------------------

    print()
    print("=" * 75)
    print("DRIFT-SENSE V5.1 - CONFIDENCE-AWARE MATCHING")
    print("=" * 75)

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

    print()

    print("V1 RESULT")
    print("-" * 75)

    print(
        f"V1 center      : "
        f"({v1_x:.2f}, {v1_y:.2f})"
    )

    print(
        f"V1 score       : "
        f"{v1_score:.4f}"
    )

    print(
        f"Second score   : "
        f"{second_score:.4f}"
    )

    print(
        f"Score gap      : "
        f"{score_gap:.4f}"
    )

    print()

    print("STRUCTURAL VERIFICATION")
    print("-" * 75)

    print(
        f"Best structural candidate : "
        f"({structural_best['center_x']:.2f}, "
        f"{structural_best['center_y']:.2f})"
    )

    print(
        f"Structural score          : "
        f"{structural_best['structural']:.4f}"
    )

    print(
        f"Structural improvement    : "
        f"{structural_improvement:.4f}"
    )

    print()

    print("FINAL DECISION")
    print("-" * 75)

    if use_v1:

        print("Using V1 prediction.")

    else:

        print(
            "Using structurally verified candidate."
        )

    print(
        f"Reason : {decision_reason}"
    )

    print()

    print(
        f"Final predicted center : "
        f"({final_x:.2f}, {final_y:.2f})"
    )

    print("=" * 75)


    return {
        "x": final_x,
        "y": final_y,
        "score": v1_score,
        "score_gap": score_gap,
        "used_v1": use_v1,
        "structural_score": structural_best["structural"],
        "structural_improvement": structural_improvement,
    }


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python localization\\baseline_v5_1.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    run_v5_1(
        sys.argv[1],
        sys.argv[2]
    )