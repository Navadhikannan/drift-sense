import sys
import cv2
import numpy as np


GT_X = 299.10
GT_Y = 618.50

# Approximate scale relationships
SCALES = [9.0, 9.5, 10.0, 10.5, 11.0]


def calculate_error(x, y):
    return np.sqrt(
        (x - GT_X) ** 2 +
        (y - GT_Y) ** 2
    )


def run_scale_sweep(reference_path, search_path):

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

    print()
    print("=" * 75)
    print("DRIFT-SENSE SCALE SWEEP - SAMPLE 001")
    print("=" * 75)

    print(
        f"Ground truth : "
        f"({GT_X:.2f}, {GT_Y:.2f})"
    )

    print()

    print(
        f"{'Scale':<10}"
        f"{'Template':<15}"
        f"{'Predicted X':<15}"
        f"{'Predicted Y':<15}"
        f"{'Score':<12}"
        f"{'Error':<12}"
    )

    print("-" * 75)

    results = []

    for scale in SCALES:

        # Search target is approximately 1/scale
        # of the original reference.
        width = max(
            1,
            int(round(reference.shape[1] / scale))
        )

        height = max(
            1,
            int(round(reference.shape[0] / scale))
        )

        template = cv2.resize(
            reference,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        result = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, location = cv2.minMaxLoc(
            result
        )

        x = (
            location[0] +
            template.shape[1] / 2.0
        )

        y = (
            location[1] +
            template.shape[0] / 2.0
        )

        error = calculate_error(x, y)

        results.append(
            {
                "scale": scale,
                "width": width,
                "height": height,
                "x": x,
                "y": y,
                "score": score,
                "error": error
            }
        )

        print(
            f"{scale:<10.1f}"
            f"{width}x{height:<10}"
            f"{x:<15.2f}"
            f"{y:<15.2f}"
            f"{score:<12.4f}"
            f"{error:<12.2f}"
        )

    print("-" * 75)

    best = min(
        results,
        key=lambda r: r["error"]
    )

    print()
    print("BEST RESULT BY GROUND-TRUTH ERROR")
    print(
        f"Scale : {best['scale']:.1f}x"
    )
    print(
        f"Prediction : "
        f"({best['x']:.2f}, {best['y']:.2f})"
    )
    print(
        f"Error : "
        f"{best['error']:.2f} px"
    )
    print(
        f"Score : "
        f"{best['score']:.4f}"
    )

    best_score = max(
        results,
        key=lambda r: r["score"]
    )

    print()
    print("BEST RESULT BY MATCHING SCORE")
    print(
        f"Scale : {best_score['scale']:.1f}x"
    )
    print(
        f"Prediction : "
        f"({best_score['x']:.2f}, {best_score['y']:.2f})"
    )
    print(
        f"Error : "
        f"{best_score['error']:.2f} px"
    )
    print(
        f"Score : "
        f"{best_score['score']:.4f}"
    )

    print("=" * 75)


if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage:"
        )

        print(
            "python localization\\scale_sweep.py "
            "<reference.png> <search.png>"
        )

        sys.exit(1)

    run_scale_sweep(
        sys.argv[1],
        sys.argv[2]
    )