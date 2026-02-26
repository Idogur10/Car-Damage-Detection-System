"""
Train YOLOv11m on ALL 6 Damage Classes
"""

from ultralytics import YOLO
import torch
import os
import shutil

# ============================================================
# CHANGE THESE SETTINGS BEFORE RUNNING
# ============================================================

EPOCHS = 100               # How many epochs to train this run
MODE = 'fresh'             # Options: 'fresh', 'continue'
                           #   'fresh'    = start over from COCO pretrained model
                           #   'continue' = load best.pt and train more epochs

# ============================================================
# DON'T CHANGE BELOW THIS LINE
# ============================================================

RUN_DIR = 'runs/detect/yolo11m_cardd_6classes'
WEIGHTS_DIR = os.path.join(RUN_DIR, 'weights')
BEST_PT = os.path.join(WEIGHTS_DIR, 'best.pt')
LAST_PT = os.path.join(WEIGHTS_DIR, 'last.pt')

CONFIG = {
    'data': 'CarDD_YOLO_6classes/data.yaml',
    'epochs': EPOCHS,
    'imgsz': 640,
    'batch': 8,
    'device': 'cuda',
    'name': RUN_DIR,
    'project': '.',
    'exist_ok': True,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'degrees': 15.0,
    'translate': 0.2,
    'scale': 0.5,
    'shear': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.5,
    'mixup': 0.0,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'patience': 100,
    'save_period': 5,
    'plots': True,
    'val': True,
}

if __name__ == '__main__':

    print("=" * 70)
    print("TRAINING YOLOv11m - ALL 6 DAMAGE CLASSES")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"Mode: {MODE}")
    print(f"Epochs: {EPOCHS}")

    if MODE == 'fresh':
        print("\nStarting fresh from COCO pretrained model...")
        model = YOLO('yolo11m.pt')

    elif MODE == 'continue':
        if not os.path.exists(BEST_PT):
            print(f"\nERROR: {BEST_PT} not found. Run with MODE='fresh' first.")
            exit(1)
        print(f"\nContinuing from: {BEST_PT}")
        print("  Using lower LR (0.001) for fine-tuning")
        model = YOLO(BEST_PT)
        CONFIG['lr0'] = 0.001
        CONFIG['warmup_epochs'] = 0

    else:
        print(f"\nERROR: Unknown MODE '{MODE}'. Use 'fresh' or 'continue'.")
        exit(1)

    results = model.train(**CONFIG)

    # Save a copy with descriptive name
    map50 = results.box.map50
    save_name = f"model_mAP{map50:.2f}.pt"
    save_path = os.path.join(WEIGHTS_DIR, save_name)
    shutil.copy2(BEST_PT, save_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  mAP50:      {map50:.4f}")
    print(f"  Precision:  {results.box.mp:.4f}")
    print(f"  Recall:     {results.box.mr:.4f}")
    print(f"\nSaved models:")
    print(f"  {BEST_PT}")
    print(f"  {LAST_PT}")
    print(f"  {save_path}  <-- copy with mAP score")
