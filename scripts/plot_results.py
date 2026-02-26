"""
Plot training results from results.csv
"""

import matplotlib.pyplot as plt
import csv
from pathlib import Path


def plot_training_results(results_csv='runs/detect/yolo11m_cardd_6classes/results.csv', output_dir='results'):
    """Plot training curves from results.csv"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Read CSV
    epochs, map50, map50_95, precision, recall = [], [], [], [], []
    train_box, train_cls, val_box, val_cls = [], [], [], []

    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            map50.append(float(row['metrics/mAP50(B)']))
            map50_95.append(float(row['metrics/mAP50-95(B)']))
            precision.append(float(row['metrics/precision(B)']))
            recall.append(float(row['metrics/recall(B)']))
            train_box.append(float(row['train/box_loss']))
            train_cls.append(float(row['train/cls_loss']))
            val_box.append(float(row['val/box_loss']))
            val_cls.append(float(row['val/cls_loss']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Results - YOLOv11m Car Damage Detection', fontsize=14, fontweight='bold')

    # mAP
    axes[0, 0].plot(epochs, map50, 'b-', label='mAP50', linewidth=2)
    axes[0, 0].plot(epochs, map50_95, 'r-', label='mAP50-95', linewidth=2)
    axes[0, 0].set_title('mAP over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Precision & Recall
    axes[0, 1].plot(epochs, precision, 'g-', label='Precision', linewidth=2)
    axes[0, 1].plot(epochs, recall, 'm-', label='Recall', linewidth=2)
    axes[0, 1].set_title('Precision & Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Training Loss
    axes[1, 0].plot(epochs, train_box, 'b-', label='Box Loss', linewidth=2)
    axes[1, 0].plot(epochs, train_cls, 'r-', label='Cls Loss', linewidth=2)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Validation Loss
    axes[1, 1].plot(epochs, val_box, 'b-', label='Box Loss', linewidth=2)
    axes[1, 1].plot(epochs, val_cls, 'r-', label='Cls Loss', linewidth=2)
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_path / 'training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == '__main__':
    plot_training_results()
