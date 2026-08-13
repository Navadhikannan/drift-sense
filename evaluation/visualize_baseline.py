import cv2
import json
import sys


def main():

    if len(sys.argv) != 4:
        print(
            "Usage: python visualize_baseline.py "
            "<search.png> <metadata.json> <output.png>"
        )
        sys.exit(1)

    search_path = sys.argv[1]
    metadata_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load search image
    image = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(search_path)

    # Convert grayscale to color
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Read ground truth
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    gt_x = float(metadata["gt_x"])
    gt_y = float(metadata["gt_y"])

    box = metadata["gt_box"]

    box_x = int(round(box[0]))
    box_y = int(round(box[1]))
    box_w = int(round(box[2]))
    box_h = int(round(box[3]))

    # Baseline prediction from our current experiment
    pred_x = 950.0
    pred_y = 562.0

    # Ground-truth box
    cv2.rectangle(
        image,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0, 255, 0),
        3
    )

    # Ground-truth center
    cv2.drawMarker(
        image,
        (int(round(gt_x)), int(round(gt_y))),
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=3
    )

    # Predicted center
    cv2.drawMarker(
        image,
        (int(round(pred_x)), int(round(pred_y))),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=3
    )

    # Labels
    cv2.putText(
        image,
        f"Ground Truth ({gt_x:.1f}, {gt_y:.1f})",
        (box_x, max(30, box_y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Prediction ({pred_x:.1f}, {pred_y:.1f})",
        (pred_x.__int__() - 180, pred_y.__int__() - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Save
    cv2.imwrite(output_path, image)

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()