"""
Evaluate Trained YOLO Model on Test/Validation Set
"""

from ultralytics import YOLO
from pathlib import Path


def evaluate_model(
    model_path='runs/detect/yolo11m_cardd_6classes/weights/best.pt',
    data_yaml='CarDD_YOLO_6classes/data.yaml',
    split='test'
):
    """
    Evaluate trained model on test/val set

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml
        split: 'val' or 'test'
    """

    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\nERROR: Model not found at: {model_path}")
        print("Please train the model first: python train_yolo11_all_classes.py")
        return

    print(f"\nModel: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Split: {split}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))

    # Run validation
    print(f"\nEvaluating on {split} set...")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        plots=True
    )

    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  mAP50:     {results.box.map50:.4f}")
    print(f"  mAP50-95:  {results.box.map:.4f}")
    print(f"  Precision:  {results.box.mp:.4f}")
    print(f"  Recall:     {results.box.mr:.4f}")

    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(model.names.values()):
        if i < len(results.box.maps):
            print(f"  {class_name}:")
            print(f"    mAP50: {results.box.maps[i]:.4f}")
            print(f"    Precision: {results.box.p[i]:.4f}")
            print(f"    Recall: {results.box.r[i]:.4f}")

    print("\n" + "="*70)
    return results


if __name__ == '__main__':
    evaluate_model()
