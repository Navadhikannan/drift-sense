import sys
import cv2
import numpy as np


# Candidate obtained from the previous candidate analysis
INITIAL_X = 449.0
INITIAL_Y = 824.0

# Search +/- 60 pixels around the candidate
RADIUS = 60

# Ground truth is used ONLY for this experiment
GT_X = 493.7
GT_Y = 823.7


def local_refine(reference_path, search_path):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(reference_path)

    if search is None:
        raise FileNotFoundError(search_path)

    # ---------------------------------------------------------
    # Same 10x scaling used by V1
    # ---------------------------------------------------------

    template = cv2.resize(
        reference,
        (100, 100),
        interpolation=cv2.INTER_AREA
    )

    h, w = search.shape
    th, tw = template.shape

    # Candidate is a CENTER coordinate.
    # Convert it to an approximate top-left coordinate.
    candidate_left = int(round(INITIAL_X - tw / 2))
    candidate_top = int(round(INITIAL_Y - th / 2))

    # ---------------------------------------------------------
    # Local search region
    # ---------------------------------------------------------

    x1 = max(0, candidate_left - RADIUS)
    y1 = max(0, candidate_top - RADIUS)

    x2 = min(w - tw, candidate_left + RADIUS)
    y2 = min(h - th, candidate_top + RADIUS)

    roi = search[
        y1:y2 + th,
        x1:x2 + tw
    ]

    # ---------------------------------------------------------
    # Template matching ONLY inside local region
    # ---------------------------------------------------------

    result = cv2.matchTemplate(
        roi,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, location = cv2.minMaxLoc(result)

    # Convert local coordinates back to full image
    refined_left = x1 + location[0]
    refined_top = y1 + location[1]

    refined_x = refined_left + tw / 2.0
    refined_y = refined_top + th / 2.0

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    error = np.sqrt(
        (refined_x - GT_X) ** 2 +
        (refined_y - GT_Y) ** 2
    )

    initial_error = np.sqrt(
        (INITIAL_X - GT_X) ** 2 +
        (INITIAL_Y - GT_Y) ** 2
    )

    improvement = initial_error - error

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print("DRIFT-SENSE LOCAL REFINEMENT")
    print("=" * 65)

    print(
        f"Initial candidate : "
        f"({INITIAL_X:.2f}, {INITIAL_Y:.2f})"
    )

    print(
        f"Initial error     : "
        f"{initial_error:.2f} px"
    )

    print()

    print(
        f"Search radius     : "
        f"+/- {RADIUS} px"
    )

    print(
        f"Refined center    : "
        f"({refined_x:.2f}, {refined_y:.2f})"
    )

    print(
        f"Matching score    : "
        f"{score:.4f}"
    )

    print(
        f"Ground truth      : "
        f"({GT_X:.2f}, {GT_Y:.2f})"
    )

    print(
        f"Refined error     : "
        f"{error:.2f} px"
    )

    print(
        f"Error improvement : "
        f"{improvement:.2f} px"
    )

    print("=" * 65)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python localization\\local_refine.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    local_refine(
        sys.argv[1],
        sys.argv[2]
    )