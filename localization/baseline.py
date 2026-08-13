import sys
import json
import cv2
import math


def localize(reference_path, search_path):
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if reference is None:
        raise FileNotFoundError(
            f"Could not read reference image: {reference_path}"
        )

    if search is None:
        raise FileNotFoundError(
            f"Could not read search image: {search_path}"
        )

    # The reference pattern appears approximately 10x smaller
    # inside the search image.
    h, w = reference.shape

    template = cv2.resize(
        reference,
        (w // 10, h // 10),
        interpolation=cv2.INTER_AREA
    )

    # Template matching
    result = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, max_location = cv2.minMaxLoc(result)

    x_top_left, y_top_left = max_location

    template_h, template_w = template.shape

    center_x = x_top_left + template_w / 2
    center_y = y_top_left + template_h / 2

    return center_x, center_y, max_score


def load_ground_truth(json_path):
    with open(json_path, "r") as f:
        metadata = json.load(f)

    gt_x = metadata["gt_x"]
    gt_y = metadata["gt_y"]

    return gt_x, gt_y


def calculate_error(pred_x, pred_y, gt_x, gt_y):
    return math.sqrt(
        (pred_x - gt_x) ** 2 +
        (pred_y - gt_y) ** 2
    )


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage: python baseline.py "
            "<reference.png> <search.png> <metadata.json>"
        )
        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]
    metadata_path = sys.argv[3]

    # Localization
    pred_x, pred_y, score = localize(
        reference_path,
        search_path
    )

    # Ground truth
    gt_x, gt_y = load_ground_truth(metadata_path)

    # Error
    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    print("=" * 50)
    print("DRIFT-SENSE BASELINE EVALUATION")
    print("=" * 50)

    print(f"Predicted center : ({pred_x:.2f}, {pred_y:.2f})")
    print(f"Ground truth     : ({gt_x:.2f}, {gt_y:.2f})")
    print(f"Localization error: {error:.2f} pixels")
    print(f"Matching score   : {score:.4f}")

    print("=" * 50)