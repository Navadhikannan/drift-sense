import sys
import cv2
import numpy as np


def find_edge_candidates(reference_path, search_path, max_candidates=30):

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
    # Resize reference to 10x smaller
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
    # Edge maps
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

    candidates = []

    temp = result.copy()

    for _ in range(max_candidates):

        _, max_score, _, max_location = cv2.minMaxLoc(temp)

        x = (
            max_location[0]
            + template.shape[1] / 2.0
        )

        y = (
            max_location[1]
            + template.shape[0] / 2.0
        )

        candidates.append(
            (x, y, float(max_score))
        )

        # Suppress neighborhood
        radius_x = max(
            10,
            template.shape[1] // 2
        )

        radius_y = max(
            10,
            template.shape[0] // 2
        )

        x1 = max(
            0,
            max_location[0] - radius_x
        )

        y1 = max(
            0,
            max_location[1] - radius_y
        )

        x2 = min(
            temp.shape[1],
            max_location[0] + radius_x + 1
        )

        y2 = min(
            temp.shape[0],
            max_location[1] + radius_y + 1
        )

        temp[y1:y2, x1:x2] = -1.0

    return candidates


def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python localization\\baseline_v4_edge_candidates.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]

    candidates = find_edge_candidates(
        reference_path,
        search_path
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    h, w = search.shape

    center_x = w / 2.0
    center_y = h / 2.0

    gt_x = 299.10
    gt_y = 618.50

    print()
    print("=" * 80)
    print("DRIFT-SENSE V4 - EDGE CANDIDATE ANALYSIS")
    print("=" * 80)

    print(
        f"{'Rank':<6}"
        f"{'X':<12}"
        f"{'Y':<12}"
        f"{'Score':<12}"
        f"{'GT Dist.':<14}"
        f"{'Center Dist.':<14}"
    )

    print("-" * 80)

    for i, (x, y, score) in enumerate(
        candidates,
        start=1
    ):

        gt_distance = np.sqrt(
            (x - gt_x) ** 2 +
            (y - gt_y) ** 2
        )

        center_distance = np.sqrt(
            (x - center_x) ** 2 +
            (y - center_y) ** 2
        )

        print(
            f"{i:<6}"
            f"{x:<12.2f}"
            f"{y:<12.2f}"
            f"{score:<12.4f}"
            f"{gt_distance:<14.2f}"
            f"{center_distance:<14.2f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()