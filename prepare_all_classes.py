"""
Prepare YOLO Data with ALL 6 Classes
Maximizes training data for better model performance
"""

import sys
sys.path.append('scripts')
from dataset_loader import CarDDDataset
from pathlib import Path
import shutil

def prepare_all_classes():
    """Prepare YOLO format data with all 6 damage classes"""

    print("="*70)
    print("PREPARING YOLO DATA - ALL 6 CLASSES")
    print("="*70)

    # All classes in CarDD
    all_classes = [
        'dent',
        'scratch',
        'crack',
        'glass shatter',
        'lamp broken',
        'tire flat'
    ]

    print(f"\nIncluding ALL damage types:")
    for i, cls in enumerate(all_classes, 1):
        print(f"  {i}. {cls}")

    output_dir = 'CarDD_YOLO_6classes'
    output_path = Path(output_dir)

    # Remove old directory if exists
    if output_path.exists():
        print(f"\nRemoving old directory: {output_dir}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} set")
        print(f"{'='*70}")

        # Load dataset
        dataset = CarDDDataset(
            'CarDD_release/CarDD_release/CarDD_COCO',
            split=split
        )

        # Show statistics
        stats = dataset.get_statistics()
        print(f"\nOriginal dataset:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")

        print(f"\nAnnotations by class:")
        for cat_name, count in stats['by_category'].items():
            print(f"  {cat_name}: {count}")

        # Export to YOLO format (NO FILTER - all classes!)
        print(f"\nExporting {split} set...")
        dataset.export_to_yolo(
            output_dir=output_dir,
            category_filter=None  # Include ALL classes!
        )

        print(f"{split.upper()} set exported")

    # Summary
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")

    print(f"\nYOLO dataset created at: {output_path.absolute()}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    images/")
    print(f"      train/    (training images)")
    print(f"      val/      (validation images)")
    print(f"      test/     (test images)")
    print(f"    labels/")
    print(f"      train/    (training labels)")
    print(f"      val/      (validation labels)")
    print(f"      test/     (test labels)")
    print(f"    data.yaml   (dataset config)")

    # Create data.yaml
    create_data_yaml(output_dir)

    # Show final statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")

    for split in ['train', 'val', 'test']:
        img_dir = output_path / 'images' / split
        lbl_dir = output_path / 'labels' / split

        n_images = len(list(img_dir.glob('*.jpg')))
        n_labels = len(list(lbl_dir.glob('*.txt')))

        print(f"\n{split.upper()}:")
        print(f"  Images: {n_images}")
        print(f"  Labels: {n_labels}")

    print(f"\n{'='*70}")
    print("Ready to train with ALL 6 classes!")
    print(f"{'='*70}")
    print(f"\nNext step: python train_yolo11_all_classes.py")

def create_data_yaml(output_dir):
    """Create data.yaml configuration file"""

    yaml_content = f"""# CarDD Dataset - All 6 Classes
# Car Damage Detection with all damage types

path: {Path(output_dir).absolute()}
train: images/train
val: images/val
test: images/test

# Classes (6 total)
names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
"""

    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated: {yaml_path}")

if __name__ == '__main__':
    prepare_all_classes()
