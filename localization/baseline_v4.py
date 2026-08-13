import sys
import cv2
import numpy as np


def find_candidates(reference_path, search_path, max_candidates=30):
    """
    V4 Candidate Analysis

    Finds multiple strong template-matching candidates
    instead of immediately accepting only the best match.
    """

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------

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
            f"Could not read reference image: {reference_path}"
        )

    if search is None:
        raise FileNotFoundError(
            f"Could not read search image: {search_path}"
        )

    # ---------------------------------------------------------
    # 10x scale relationship
    # Reference is approximately 10x larger than target
    # ---------------------------------------------------------

    template_size = (
        max(1, search.shape[1] // 10),
        max(1, search.shape[0] // 10)
    )

    template = cv2.resize(
        reference,
        template_size,
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------------------------
    # Template matching
    # ---------------------------------------------------------

    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    # ---------------------------------------------------------
    # Find local maxima
    # ---------------------------------------------------------

    candidates = []

    temp = result.copy()

    for _ in range(max_candidates):

        _, max_value, _, max_location = cv2.minMaxLoc(temp)

        x = max_location[0]
        y = max_location[1]

        center_x = x + template.shape[1] / 2.0
        center_y = y + template.shape[0] / 2.0

        candidates.append({
            "x": center_x,
            "y": center_y,
            "score": float(max_value)
        })

        # -----------------------------------------------------
        # Suppress an area around the current maximum
        # so the next candidate is a different region.
        # -----------------------------------------------------

        radius_x = max(10, template.shape[1] // 2)
        radius_y = max(10, template.shape[0] // 2)

        x1 = max(0, x - radius_x)
        y1 = max(0, y - radius_y)

        x2 = min(
            temp.shape[1],
            x + radius_x + 1
        )

        y2 = min(
            temp.shape[0],
            y + radius_y + 1
        )

        temp[y1:y2, x1:x2] = -1.0

    return candidates


def print_candidates(
    reference_path,
    search_path
):

    candidates = find_candidates(
        reference_path,
        search_path,
        max_candidates=30
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    search_h, search_w = search.shape

    center_x = search_w / 2.0
    center_y = search_h / 2.0

    print()
    print("=" * 70)
    print("DRIFT-SENSE V4 - CANDIDATE ANALYSIS")
    print("=" * 70)

    print(
        f"Search size : "
        f"{search_w} x {search_h}"
    )

    print(
        f"Candidates found : "
        f"{len(candidates)}"
    )

    print()
    print(
        f"{'Rank':<6}"
        f"{'X':<12}"
        f"{'Y':<12}"
        f"{'Score':<12}"
        f"{'Center Dist.':<15}"
    )

    print("-" * 70)

    for i, candidate in enumerate(
        candidates,
        start=1
    ):

        x = candidate["x"]
        y = candidate["y"]
        score = candidate["score"]

        distance = np.sqrt(
            (x - center_x) ** 2 +
            (y - center_y) ** 2
        )

        print(
            f"{i:<6}"
            f"{x:<12.2f}"
            f"{y:<12.2f}"
            f"{score:<12.4f}"
            f"{distance:<15.2f}"
        )

    print("=" * 70)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python localization\\baseline_v4.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]

    print_candidates(
        reference_path,
        search_path
    )