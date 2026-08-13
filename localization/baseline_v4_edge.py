import sys
import cv2
import numpy as np


def edge_match(reference_path, search_path):

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
    # Resize reference to approximately 10x smaller
    # ---------------------------------------------------------

    template = cv2.resize(
        reference,
        (
            search.shape[1] // 10,
            search.shape[0] // 10
        ),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------------------------
    # Edge extraction
    # ---------------------------------------------------------

    template_edges = cv2.Canny(
        template,
        50,
        150
    )

    search_edges = cv2.Canny(
        search,
        50,
        150
    )

    # ---------------------------------------------------------
    # Edge template matching
    # ---------------------------------------------------------

    result = cv2.matchTemplate(
        search_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, max_location = cv2.minMaxLoc(
        result
    )

    x = (
        max_location[0]
        + template.shape[1] / 2.0
    )

    y = (
        max_location[1]
        + template.shape[0] / 2.0
    )

    return x, y, max_score


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python localization\\baseline_v4_edge.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]

    x, y, score = edge_match(
        reference_path,
        search_path
    )

    print()
    print("=" * 60)
    print("DRIFT-SENSE V4 - EDGE MATCHING")
    print("=" * 60)

    print(
        f"Predicted center : "
        f"({x:.2f}, {y:.2f})"
    )

    print(
        f"Matching score   : "
        f"{score:.4f}"
    )

    print("=" * 60)