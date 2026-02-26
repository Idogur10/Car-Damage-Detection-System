# Car Damage Detection System

Object detection system that identifies and localizes vehicle damage using **YOLOv11m** with transfer learning from COCO.

Trained on the [CarDD dataset](https://cardd-ustc.github.io/) (4,000 images, 6 damage classes).

---

## Results

**Best model performance (100 epochs):**

| Metric | Score |
|--------|-------|
| mAP50 | 0.737 |
| mAP50-95 | 0.577 |
| Precision | 0.776 |
| Recall | 0.695 |

**Per-class detection (6 classes):**

| Class | Annotations | Notes |
|-------|-------------|-------|
| scratch | 3,595 | Most common |
| dent | 2,543 | |
| crack | 898 | |
| lamp broken | 704 | |
| glass shatter | 681 | |
| tire flat | 319 | Least data |

---

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Train (fresh)
python train_yolo11_all_classes.py

# Run detection on an image
python scripts/inference.py car_image.jpg
```

---

## Project Structure

```
train_yolo11_all_classes.py       # Main training script
prepare_all_classes.py            # Convert CarDD to YOLO format
scripts/
    dataset_loader.py             # Dataset loading & conversion
    evaluate_model.py             # Evaluate on test/val set
    inference.py                  # Run detection on new images
    visualize_results.py          # Prediction visualizations
    plot_results.py               # Training curve plots
docs/                             # Technical documentation
runs/detect/yolo11m_cardd_6classes/
    weights/best.pt               # Best trained model
    results.csv                   # Training metrics
    results.png                   # Training curves
    confusion_matrix.png          # Confusion matrix
    val_batch*_pred.jpg           # Validation predictions
```

---

## Training

### 1. Prepare Dataset

Download the [CarDD dataset](https://cardd-ustc.github.io/) and place in `CarDD_release/`, then:

```bash
python prepare_all_classes.py
```

### 2. Train

Edit `train_yolo11_all_classes.py`:
- `MODE = 'fresh'` to start from scratch
- `MODE = 'continue'` to fine-tune from best.pt (uses lower LR)
- `EPOCHS = 100` to set epoch count

```bash
python train_yolo11_all_classes.py
```

**Training config:**
- Model: YOLOv11m (20M params, COCO pretrained)
- Image size: 640px
- Batch size: 8
- Optimizer: SGD (lr=0.01 fresh, lr=0.001 continue)
- Augmentation: mosaic, horizontal flip, rotation, translation, scale
- GPU: RTX 3060 6GB (~2 min/epoch)

### 3. Evaluate

```bash
python scripts/evaluate_model.py
```

---

## Inference

**Python:**
```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11m_cardd_6classes/weights/best.pt')
results = model('car_image.jpg', conf=0.25)
results[0].show()
```

**CLI:**
```bash
python scripts/inference.py car_image.jpg
python scripts/inference.py car_photos/ --batch
python scripts/inference.py car_image.jpg --conf 0.5
```

---

## Dataset

**CarDD (Car Damage Detection)** - 4,000 images, 9,740 annotations

| Split | Images | Size |
|-------|--------|------|
| Train | 2,816 | 2.0 GB |
| Val | 810 | 592 MB |
| Test | 374 | 272 MB |

---

## Documentation

Technical docs in `docs/`:

- [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) - Project guide
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training instructions
- [TRAINING_PROCESS_EXPLAINED.md](docs/TRAINING_PROCESS_EXPLAINED.md) - How YOLO training works
- [TRANSFER_LEARNING_EXPLAINED.md](docs/TRANSFER_LEARNING_EXPLAINED.md) - Transfer learning concepts
- [LOSS_AND_OPTIMIZER_EXPLAINED.md](docs/LOSS_AND_OPTIMIZER_EXPLAINED.md) - Loss functions & optimization
- [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) - YOLO model sizes
- [YOLO_VERSION_COMPARISON.md](docs/YOLO_VERSION_COMPARISON.md) - YOLOv8 vs YOLOv11

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA
- 6GB+ VRAM

```bash
pip install -r requirements.txt
```

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO framework
- [CarDD Dataset](https://cardd-ustc.github.io/) - Car damage dataset
