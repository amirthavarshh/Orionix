import os
import shutil
import random
import subprocess
from pathlib import Path
from PIL import Image

def main():
    dataset_name = "nilesh789/eurosat-satellite-image-classification"
    base_dir = Path("backend/ml/data/eurosat")
    raw_dir = base_dir / "raw"
    
    print("Checking Kaggle installation and credentials...")
    try:
        import kaggle
    except ImportError:
        print("Warning: kaggle python package not found in current environment. Using kaggle CLI if available.")

    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_config_dir / "kaggle.json"
    if not kaggle_json_path.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print(f"\n[ERROR] Kaggle credentials not found at {kaggle_json_path}")
        print("Please authenticate with Kaggle by placing your kaggle.json in ~/.kaggle/ ")
        print("or setting KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {dataset_name} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(raw_dir), "--unzip"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        return
    except FileNotFoundError:
        print("\n[ERROR] kaggle CLI not found. Please install kaggle (pip install kaggle) and ensure it's in your PATH.")
        return

    print("\nLocating class directories in the extracted dataset...")
    class_dirs = []
    search_dirs = [raw_dir]
    while search_dirs:
        curr_dir = search_dirs.pop(0)
        subdirs = [d for d in curr_dir.iterdir() if d.is_dir()]
        if len(subdirs) >= 10: # EuroSAT has 10 classes
            class_dirs = subdirs
            break
        search_dirs.extend(subdirs)
        
    if not class_dirs:
        print("[ERROR] Could not find class directories in the extracted dataset.")
        return

    print(f"Found {len(class_dirs)} class directories. Creating train/val/test splits (70/15/15)...")
    splits = ["train", "val", "test"]
    for split in splits:
        for class_dir in class_dirs:
            (base_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    distribution = {split: {class_dir.name: 0 for class_dir in class_dirs} for split in splits}
    sample_shape = None

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]
        images.sort(key=lambda x: x.name) # Ensure reproducible order before shuffling
        random.shuffle(images)
        
        if images and sample_shape is None:
            with Image.open(images[0]) as img:
                sample_shape = img.size # (width, height)

        n = len(images)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for img in train_imgs:
            shutil.copy(img, base_dir / "train" / class_name / img.name)
            distribution["train"][class_name] += 1
            
        for img in val_imgs:
            shutil.copy(img, base_dir / "val" / class_name / img.name)
            distribution["val"][class_name] += 1
            
        for img in test_imgs:
            shutil.copy(img, base_dir / "test" / class_name / img.name)
            distribution["test"][class_name] += 1

    print("\nDataset Split Complete.")
    print(f"Sample image dimensions (Width x Height): {sample_shape}")
    print("\nClass Distribution:")
    for split in splits:
        total = sum(distribution[split].values())
        print(f"\n{split.upper()} ({total} total):")
        for cls, count in distribution[split].items():
            print(f"  {cls}: {count}")

    # Generate dataset_summary.md
    summary_path = base_dir / "dataset_summary.md"
    with open(summary_path, "w") as f:
        f.write("# EuroSAT Dataset Summary\n\n")
        f.write(f"Dataset Source: `kaggle datasets download -d {dataset_name}`\n")
        if sample_shape:
            f.write(f"Image Dimensions: {sample_shape[0]}x{sample_shape[1]} pixels\n")
        f.write("Split Ratio: 70% Train, 15% Validation, 15% Test (Random Seed: 42)\n\n")
        f.write("## Class Distribution\n\n")
        f.write("| Class | Train (70%) | Validation (15%) | Test (15%) | Total |\n")
        f.write("|---|---|---|---|---|\n")
        
        total_train, total_val, total_test = 0, 0, 0
        for cls in sorted([d.name for d in class_dirs]):
            train_c = distribution["train"][cls]
            val_c = distribution["val"][cls]
            test_c = distribution["test"][cls]
            total_c = train_c + val_c + test_c
            f.write(f"| {cls} | {train_c} | {val_c} | {test_c} | {total_c} |\n")
            total_train += train_c
            total_val += val_c
            total_test += test_c
            
        overall_total = total_train + total_val + total_test
        f.write(f"| **Total** | **{total_train}** | **{total_val}** | **{total_test}** | **{overall_total}** |\n")
    
    print(f"\nSummary successfully saved to {summary_path}")

if __name__ == "__main__":
    main()
