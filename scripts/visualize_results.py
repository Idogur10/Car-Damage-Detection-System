"""
Visualize model predictions vs ground truth on test images
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os


CLASS_NAMES = {0: 'dent', 1: 'scratch', 2: 'crack', 3: 'glass shatter', 4: 'lamp broken', 5: 'tire flat'}
CLASS_COLORS = {
    0: (255, 0, 0),      # Red - dent
    1: (0, 255, 0),      # Green - scratch
    2: (0, 0, 255),      # Blue - crack
    3: (255, 255, 0),    # Cyan - glass shatter
    4: (255, 0, 255),    # Magenta - lamp broken
    5: (0, 165, 255),    # Orange - tire flat
}


def load_yolo_labels(label_path):
    """Load YOLO format labels"""
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                labels.append({
                    'class_id': int(class_id),
                    'bbox': [x_center, y_center, width, height]
                })
    return labels


def denormalize_bbox(bbox, img_width, img_height):
    """Convert normalized YOLO bbox to pixel coordinates"""
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2


def visualize_predictions(
    model_path='runs/detect/yolo11m_cardd_6classes/weights/best.pt',
    data_yaml='CarDD_YOLO_6classes/data.yaml',
    output_dir='results',
    num_samples=12
):
    """Create visualizations of model predictions vs ground truth"""

    print("=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model = YOLO(model_path)

    test_images_dir = Path('CarDD_YOLO_6classes/images/test')
    test_labels_dir = Path('CarDD_YOLO_6classes/labels/test')
    test_images = sorted(list(test_images_dir.glob('*.jpg')))[:num_samples]

    print(f"Visualizing {len(test_images)} test images...")

    n_cols = 3
    n_rows = (len(test_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(test_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]

        label_path = test_labels_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_labels(label_path)

        results = model(str(img_path), conf=0.25, verbose=False)

        # Draw ground truth boxes
        for label in gt_labels:
            x1, y1, x2, y2 = denormalize_bbox(label['bbox'], img_width, img_height)
            class_id = label['class_id']
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"GT: {CLASS_NAMES.get(class_id, '?')}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw predictions
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(img, f"{CLASS_NAMES.get(class_id, '?')} {conf:.2f}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        ax.imshow(img)
        ax.set_title(f"{img_path.name}\nGT: {len(gt_labels)} | Pred: {len(results[0].boxes)}", fontsize=10)
        ax.axis('off')

    for idx in range(len(test_images), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    output_file = output_path / 'predictions_vs_groundtruth.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

    print("\nDone!")
    return output_path


if __name__ == '__main__':
    visualize_predictions()
