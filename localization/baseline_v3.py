import sys
import json
import math
import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def load_ground_truth(path):
    with open(path, "r") as f:
        metadata = json.load(f)

    return float(metadata["gt_x"]), float(metadata["gt_y"])


def calculate_error(px, py, gx, gy):
    return math.sqrt(
        (px - gx) ** 2 +
        (py - gy) ** 2
    )


def localize(reference, search):

    # --------------------------------------------------------
    # The reference is 10x higher magnification.
    # Convert it to approximately the same scale as the
    # target inside the search image.
    # --------------------------------------------------------

    reference_small = cv2.resize(
        reference,
        (reference.shape[1] // 10,
         reference.shape[0] // 10),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # SIFT feature detector
    # --------------------------------------------------------

    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints_ref, descriptors_ref = sift.detectAndCompute(
        reference_small,
        None
    )

    keypoints_search, descriptors_search = sift.detectAndCompute(
        search,
        None
    )

    if descriptors_ref is None:
        raise RuntimeError(
            "No SIFT features found in reference image."
        )

    if descriptors_search is None:
        raise RuntimeError(
            "No SIFT features found in search image."
        )

    # --------------------------------------------------------
    # Feature matching
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    matches = matcher.knnMatch(
        descriptors_ref,
        descriptors_search,
        k=2
    )

    # Lowe ratio test
    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) == 0:
        raise RuntimeError(
            "No reliable feature matches found."
        )

    # --------------------------------------------------------
    # Convert matched search points into coordinates
    # --------------------------------------------------------

    points = []

    for match in good_matches:

        search_point = keypoints_search[
            match.trainIdx
        ].pt

        points.append(search_point)

    points = np.asarray(points, dtype=np.float32)

    # --------------------------------------------------------
    # Robust location estimation
    #
    # Repetitive structures can create many matches.
    # We therefore use a robust median estimate first.
    # --------------------------------------------------------

    predicted_x = float(np.median(points[:, 0]))
    predicted_y = float(np.median(points[:, 1]))

    return (
        predicted_x,
        predicted_y,
        len(keypoints_ref),
        len(keypoints_search),
        len(good_matches)
    )


def main():

    if len(sys.argv) != 4:

        print(
            "Usage:\n"
            "python baseline_v3.py "
            "<reference.png> <search.png> <metadata.json>"
        )

        sys.exit(1)

    reference_path = sys.argv[1]
    search_path = sys.argv[2]
    metadata_path = sys.argv[3]

    reference = load_image(reference_path)
    search = load_image(search_path)

    (
        pred_x,
        pred_y,
        ref_features,
        search_features,
        good_matches
    ) = localize(reference, search)

    gt_x, gt_y = load_ground_truth(metadata_path)

    error = calculate_error(
        pred_x,
        pred_y,
        gt_x,
        gt_y
    )

    print("=" * 60)
    print("DRIFT-SENSE BASELINE V3 - SIFT")
    print("=" * 60)

    print(
        f"Reference size       : "
        f"{reference.shape[1]} x {reference.shape[0]}"
    )

    print(
        f"Search size          : "
        f"{search.shape[1]} x {search.shape[0]}"
    )

    print(
        f"Scaled reference     : "
        f"{reference.shape[1] // 10} x "
        f"{reference.shape[0] // 10}"
    )

    print()
    print(f"Reference features   : {ref_features}")
    print(f"Search features      : {search_features}")
    print(f"Good feature matches : {good_matches}")

    print()
    print(
        f"Predicted center     : "
        f"({pred_x:.2f}, {pred_y:.2f})"
    )

    print(
        f"Ground truth         : "
        f"({gt_x:.2f}, {gt_y:.2f})"
    )

    print(
        f"Localization error   : "
        f"{error:.2f} pixels"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()